"""RetinaFraktal — 2204-A için araştırma amaçlı Streamlit uygulaması.

Girdi: tercihen uzman tarafından işaretlenmiş ikili damar maskesi.
Çıktı: çoklu ızgara kutu sayımı ve koşula bağlı ölçek aralığı karşılaştırması.
Tanı / klinik doğruluk aracı değildir.
"""
import io
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize

st.set_page_config(page_title="RetinaFraktal", page_icon="🔬", layout="wide")
st.title("RetinaFraktal | Kutu Sayma ve Ölçek Kararlılığı")
st.caption("TÜBİTAK 2204-A araştırma prototipi • Klinik tanı amacıyla kullanılamaz")


def example_mask(n=512):
    """Sadece yazılımı denemek için üretilen sentetik dallanma; retina değildir."""
    im = Image.new("L", (n, n), 0)
    dr = ImageDraw.Draw(im)
    def branch(x, y, length, angle, depth):
        if depth <= 0 or length < 3:
            return
        a = np.deg2rad(angle)
        x2 = x + length * np.sin(a)
        y2 = y - length * np.cos(a)
        dr.line((int(x), int(y), int(x2), int(y2)), fill=255, width=max(1, depth // 2))
        branch(x2, y2, length * 0.74, angle - 23, depth - 1)
        branch(x2, y2, length * 0.73, angle + 27, depth - 1)
    branch(n / 2, n - 25, n / 4, 0, 8)
    return np.asarray(im) > 0


def mask_from_upload(uploaded, threshold, invert):
    im = Image.open(uploaded).convert("L")
    g = np.asarray(im)
    mask = g >= threshold
    if invert:
        mask = ~mask
    return mask


def box_count(mask, size, offset_y, offset_x):
    """Kareleri sonsuz ızgaranın parçası kabul eder; kenardaki tüm pikseller dahildir.
    Izgara indisleri floor((koordinat + offset)/size) şeklindedir.
    """
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return 0
    cols = (xx + offset_x) // size
    rows = (yy + offset_y) // size
    # 2D kutu kimliklerini hızlıca eşsizleştir
    stride = (mask.shape[1] + offset_x) // size + 2
    ids = rows.astype(np.int64) * stride + cols.astype(np.int64)
    return int(np.unique(ids).size)


def fit_dimension(scales, counts, image_width):
    """log(N) = D * log(W/s) + a; D eğimdir."""
    scales = np.asarray(scales, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if len(scales) < 3 or np.any(counts <= 0):
        return np.nan, np.nan
    x = np.log(image_width / scales)
    y = np.log(counts)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    denom = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(residual ** 2) / denom if denom > 1e-12 else np.nan
    return float(slope), float(r2)


def draw_grid(mask, size, fy, fx):
    h, w = mask.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mask, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    oy, ox = int(fy * size), int(fx * size)
    for x in np.arange(-ox, w + size, size):
        ax.axvline(x - 0.5, lw=0.5, color="orange", alpha=0.85)
    for y in np.arange(-oy, h + size, size):
        ax.axhline(y - 0.5, lw=0.5, color="orange", alpha=0.85)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_title(f"Kutu: {size}px | Kaydırma: ({fx:.2f}, {fy:.2f})")
    ax.set_axis_off()
    return fig


st.sidebar.header("1 • Görüntü ve maske")
upload = st.sidebar.file_uploader("Damar maskesi yükle (PNG, JPG, TIFF, PPM)", type=["png", "jpg", "jpeg", "tif", "tiff", "ppm"])
threshold = st.sidebar.slider("İkili maske eşik değeri", 0, 255, 128)
invert = st.sidebar.checkbox("Siyah damar / beyaz arka plan (ters çevir)", value=False)
thin = st.sidebar.checkbox("İskeletleştirme (1 piksel kalınlık)", value=True)

if upload is None:
    st.info("Dosya yüklenmedi: yazılımı göstermek için sentetik dallanma örüntüsü kullanılıyor. Gerçek çalışmada doğrulanmış damar maskesi yükleyin.")
    mask = example_mask()
else:
    mask = mask_from_upload(upload, threshold, invert)
if thin and mask.any():
    mask = skeletonize(mask)
mask = np.asarray(mask, dtype=bool)
h, w = mask.shape
if not mask.any() or mask.all():
    st.error("Maske tümüyle boş veya dolu. Doğru eşik ve ters çevirme seçeneğini kontrol edin.")
    st.stop()

st.sidebar.header("2 • Deney ayarları")
avail = [2 ** k for k in range(1, 11) if 2 ** k <= min(w, h) // 2]
if len(avail) < 4:
    st.error("Görüntü çok küçük. En az dört farklı 2'nin kuvveti kutu ölçeği gerekli.")
    st.stop()
chosen = st.sidebar.multiselect("Kutu kenarları (piksel)", avail, default=[x for x in [4, 8, 16, 32, 64, 128] if x in avail])
chosen = sorted(chosen)
window = st.sidebar.slider("Bir ölçek aralığındaki ölçek sayısı", 3, max(3, len(chosen)), min(4, len(chosen))) if len(chosen)>=3 else 4
quarters = st.sidebar.selectbox("Her eksende kaydırma oranı sayısı", [2, 4], index=1,
                              help="4 seçilirse 0, 0.25, 0.50, 0.75; iki eksende 16 konum")
min_r2 = st.sidebar.slider("Öneri için asgari ortalama R²", 0.90, 1.0, 0.98, 0.005)

c1, c2 = st.columns([1, 2])
with c1:
    st.image(mask.astype(np.uint8) * 255, caption=f"Analiz edilen maske • {w}×{h} piksel", clamp=True)
    st.metric("Damar içeren piksel", f"{mask.sum():,}")
with c2:
    st.markdown("**Yöntem**: Her kutu büyüklüğünde bütün damar pikselleri sayılır. Izgara, kutu kenarının belirli kesirleri kadar iki eksende kaydırılır. Her konum ve ölçek aralığı için log-log regresyonunun eğimi hesaplanır.")
    st.warning("Renkli fundus fotoğrafını doğrudan yüklemek geçerli damar analizi sağlamaz. Tercihen uzman tarafından hazırlanmış ikili damar maskesi kullanın. Eşikleme ve iskeletleştirme sonuçları değiştirebilir.")

if len(chosen) < 3:
    st.warning("Lütfen en az 3 kutu büyüklüğü seçin.")
    st.stop()

fractions = np.arange(quarters, dtype=float) / quarters
positions = [(float(fy), float(fx)) for fy in fractions for fx in fractions]
windows = [tuple(chosen[i:i + window]) for i in range(len(chosen)-window+1)]

if st.button("Analizi çalıştır", type="primary"):
    with st.spinner("Kutu sayımları ve regresyonlar hesaplanıyor..."):
        raw = []
        start = time.perf_counter()
        for j, (fy, fx) in enumerate(positions):
            for size in chosen:
                t0 = time.perf_counter()
                cnt = box_count(mask, size, int(fy * size), int(fx * size))
                elapsed = time.perf_counter() - t0
                raw.append({"konum": j + 1, "y_oran": fy, "x_oran": fx,
                            "kutu_px": size, "dolu_kutu": cnt, "sure_s": elapsed})
        raw_df = pd.DataFrame(raw)
        estimates = []
        for scales in windows:
            label = f"{scales[0]}–{scales[-1]} px"
            for j in range(len(positions)):
                part = raw_df[(raw_df.konum == j+1) & (raw_df.kutu_px.isin(scales))]
                part = part.set_index("kutu_px").loc[list(scales)].reset_index()
                dim, r2 = fit_dimension(part.kutu_px, part.dolu_kutu, w)
                estimates.append({"aralik": label, "konum": j+1, "D": dim, "R2": r2,
                                  "sure_s": float(part.sure_s.sum())})
        est_df = pd.DataFrame(estimates)
        summary = est_df.groupby("aralik", sort=False).agg(
            ort_D=("D", "mean"), sd_D=("D", "std"), ort_R2=("R2", "mean"),
            ort_sure_s=("sure_s", "mean"), n_konum=("konum", "count")
        ).reset_index()
        summary["CV_yuzde"] = 100 * summary.sd_D / summary.ort_D.abs()
        summary["uygun_aday"] = (summary.ort_R2 >= min_r2) & (summary.ort_D.between(0, 2))
        valid = summary[summary.uygun_aday & summary.CV_yuzde.notna()]
        selected = None
        if not valid.empty:
            # Araştırma amaçlı BASİT kural: önce düşük CV, eşitlikte kısa süre
            selected = str(valid.sort_values(["CV_yuzde", "ort_sure_s"]).iloc[0].aralik)
        st.session_state["results"] = (raw_df, est_df, summary, selected)
        st.session_state["setting"] = (tuple(chosen), window, quarters, threshold, invert, thin, w, h, upload.name if upload else "sentetik")
        st.session_state["total_time"] = time.perf_counter() - start

if "results" in st.session_state:
    raw_df, est_df, summary, selected = st.session_state["results"]
    st.subheader("Analiz sonuçları")
    st.caption("Sonuçlar son çalıştırmadan alınmıştır. Ayarları değiştirirseniz analizi yeniden çalıştırın.")
    a, b, c = st.columns(3)
    a.metric("Izgara konumu", len(positions))
    b.metric("Ölçek aralığı", len(windows))
    c.metric("Son analizin hesaplama süresi", f"{st.session_state['total_time']:.3f} s")
    st.dataframe(summary.round(5), use_container_width=True, hide_index=True)
    if selected:
        st.success(f"Deneysel karar kuralının önerdiği aralık: {selected}. Bu sonuç yalnızca tanımlanan ölçütlerde en düşük CV'yi gösterir; gerçek fraktal boyuta yakınlığı kanıtlamaz.")
    else:
        st.warning("Belirlenen R² ve boyut sınırlarını sağlayan aralık yok. Eşiği veya ölçekleri gözden geçirin.")
    st.markdown("#### Izgara konumlarına göre fraktal boyut")
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, part in est_df.groupby("aralik", sort=False):
        ax.plot(part.konum, part.D, marker="o", label=label)
    ax.set(xlabel="Izgara konumu", ylabel="Tahmini fraktal boyut")
    ax.legend()
    ax.grid(alpha=.25)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Izgara üst üste gösterimi")
    vis_scale = st.selectbox("Görselleştirilecek kutu kenarı", chosen, index=min(3, len(chosen)-1))
    vis_pos = st.selectbox("Izgara konumu", list(range(1, len(positions)+1)))
    fy, fx = positions[vis_pos-1]
    fig = draw_grid(mask, vis_scale, fy, fx)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Ham veriler")
    st.dataframe(est_df.round(6), use_container_width=True, hide_index=True)
    st.download_button("Kutu sayımları CSV", raw_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="kutu_sayimlari.csv", mime="text/csv")
    st.download_button("Boyut tahminleri CSV", est_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="boyut_tahminleri.csv", mime="text/csv")
    st.download_button("Özet CSV", summary.to_csv(index=False).encode("utf-8-sig"),
                       file_name="ozet.csv", mime="text/csv")
    st.info("**Bilimsel sınırlılık:** 16 kaydırma aynı retina görüntüsündeki ilişkili ölçümlerdir. Düşük CV klinik doğruluk değildir. Ölçek seçim kuralının eşikleri geliştirme kümesinde belirlenip ayrı görüntülerde doğrulanmalıdır. İşlem süreleri donanım ve çalışma koşullarına bağlıdır.")
