import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
import gdown 
import h5py
import json
# import time

# --- Modern CSS Bootstrap-like ----
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #f8fafc 0%, #e0ecfa 100%);
    color: #000000;
}
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #e0ecfa 100%);
    color: #000000;
}

/* Default warna teks elemen umum – TIDAK lagi memaksa div & span */
body, p, li, small, h1, h2, h3, h4, h5, h6 {
    color: #000000;
}

.card {
    background: #fff;
    border-radius: 2rem;
    box-shadow: 0 4px 32px 0 rgba(99, 154, 209, 0.11), 0 1.5px 7px 0 rgba(130, 160, 190, 0.07);
    padding: 2.2rem 2.2rem 1.5rem 2.2rem;
    margin-bottom: 1.5rem;
    transition: 0.3s all;
}
.card:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 6px 44px 0 rgba(99, 154, 209, 0.17), 0 2.5px 11px 0 rgba(130, 160, 190, 0.13);
}
.centered {
    display: flex;
    flex-direction: column;
    align-items: center;
}
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 700;
}
.animate-fadein {
    animation: fadeIn 1.3s;
}
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(12px);}
    100% { opacity: 1; transform: translateY(0);}
}
.upload-btn {
    background: linear-gradient(90deg, #52a2ff 0%, #7ddc92 100%);
    color: white !important;
    border: none;
    border-radius: 2rem;
    padding: 0.6rem 2.2rem;
    font-size: 1.1rem;
    box-shadow: 0 3px 8px 0 rgba(80, 132, 175, 0.15);
    transition: 0.22s all;
    cursor: pointer;
}
.upload-btn:hover {
    background: linear-gradient(90deg, #3c85f3 0%, #2fbc77 100%);
    box-shadow: 0 6px 20px 0 rgba(99, 154, 209, 0.14);
    transform: scale(1.045);
}
.stProgress > div > div > div > div {
    background-image: linear-gradient(90deg, #2db6f8 10%, #57e299 100%);
}
::-webkit-scrollbar-thumb {background: #e5eaf1;}
::-webkit-scrollbar {width: 6px;}

/* --- Loader karakter AI --- */
.loader-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.5rem 0 0.3rem 0;
}

.loader-pulse {
    width: 130px;
    height: 130px;
    border-radius: 50%;
    border: 2px dashed #2db6f8;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: pulse 1.8s ease-out infinite;
    background: radial-gradient(circle at 30% 30%, #ffffff, #e5f4ff);
}

.loader-avatar {
    width: 96px;
    height: 96px;
    border-radius: 50%;
    background: #e0f4ff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 46px;
    animation: float 2.3s ease-in-out infinite;
}

.loader-caption {
    margin-top: 0.6rem;
    font-size: 0.9rem;
    color: #000000;
    text-align: center;
    max-width: 260px;
}

@keyframes float {
    0%   { transform: translateY(0px); }
    50%  { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}

@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(45,182,248,0.45); }
    70%  { box-shadow: 0 0 0 22px rgba(45,182,248,0); }
    100% { box-shadow: 0 0 0 0 rgba(45,182,248,0); }
}

/* --- Styling tombol Streamlit umum (button & download) --- */
.stButton > button,
.stDownloadButton > button {
    color: #ffffff !important;
    background: linear-gradient(90deg, #52a2ff 0%, #7ddc92 100%) !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 0.55rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    box-shadow: 0 3px 10px rgba(80, 132, 175, 0.25) !important;
    transition: 0.22s all ease-in-out !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 6px 22px rgba(80, 132, 175, 0.35) !important;
}

/* --- Styling khusus untuk tombol Browse file di uploader --- */
.stFileUploader div[role="button"],
.stFileUploader button,
.stFileUploader label,
.stFileUploader span {
    color: #ffffff !important;
}

.stFileUploader div[role="button"],
.stFileUploader button {
    background: linear-gradient(90deg, #52a2ff 0%, #7ddc92 100%) !important;
    border-radius: 999px !important;
    border: none !important;
    padding: 0.55rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    box-shadow: 0 3px 10px rgba(80, 132, 175, 0.25) !important;
    transition: 0.22s all ease-in-out !important;
}

.stFileUploader div[role="button"]:hover,
.stFileUploader button:hover {
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 6px 22px rgba(80, 132, 175, 0.35) !important;
}

/* --- Bikin area putih (dropzone) lebih berguna dengan teks bantuan --- */
.stFileUploader > div > div:nth-child(2) {
    position: relative;
    border-radius: 999px !important;
    border: 1px dashed rgba(148, 163, 184, 0.7) !important;
    background: #ffffff !important;
    min-height: 52px !important;
}

.stFileUploader > div > div:nth-child(2)::before {
    content: "Tarik & letakkan citra X-ray di sini atau klik untuk memilih / mengganti file.";
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 0.85rem;
    font-weight: 500;
    color: #9ca3af;
    pointer-events: none;
}

/* --- Tombol link ke contoh gambar (styling mirip tombol lain) --- */
.gradient-link-btn {
    display: inline-block;
    background: linear-gradient(90deg, #52a2ff 0%, #7ddc92 100%);
    color: #ffffff !important;
    border-radius: 999px;
    padding: 0.55rem 1.5rem;
    font-weight: 600;
    font-size: 0.95rem;
    text-decoration: none !important;
    box-shadow: 0 3px 10px rgba(80, 132, 175, 0.25);
    transition: 0.22s all ease-in-out;
}
.gradient-link-btn:hover {
    transform: translateY(-2px) scale(1.03);
    box-shadow: 0 6px 22px rgba(80, 132, 175, 0.35);
}
</style>
""", unsafe_allow_html=True)

# ---- Konstanta & Label
IMG_HEIGHT, IMG_WIDTH = 224, 224
LABELS = [
    'Grade 0 (Normal)', 
    'Grade 1 (Doubtful)', 
    'Grade 2 (Mild)', 
    'Grade 3 (Moderate)', 
    'Grade 4 (Severe)'
]

# ---- Penjelasan ilmiah per KL Grade (dengan sitasi)
KETERANGAN = [
    (
        "Tidak tampak kelainan radiografis yang bermakna pada sendi lutut: "
        "celah sendi dipertahankan, tidak terlihat osteofit, sklerosis subkondral, "
        "maupun deformitas tulang. Temuan ini konsisten dengan definisi KL Grade 0, "
        "yaitu sendi yang dianggap radiografis normal tanpa tanda osteoartritis "
        "[5], [4]."
    ),
    (
        "Terlihat osteofit yang sangat kecil atau meragukan di sekitar kompartemen "
        "tibiofemoral dan/atau femoropatellar, tanpa penyempitan celah sendi yang "
        "jelas. Kondisi ini dikategorikan sebagai KL Grade 1, yang sering dianggap "
        "sebagai fase sangat awal atau ‘doubtful’ osteoartritis di mana perubahan "
        "degeneratif pada kartilago dan tulang subkondral baru mulai muncul [5], [4]."
    ),
    (
        "Terdapat osteofit yang jelas (definitif) dengan kemungkinan penyempitan "
        "ringan pada celah sendi, terutama pada kompartemen yang menanggung beban. "
        "Secara radiografis, kategori ini sesuai dengan KL Grade 2, yang mengindikasikan "
        "osteoartritis derajat ringan, sering disertai gejala klinis seperti nyeri "
        "dan kaku lutut yang intermittent [5], [4], [7]–[9]."
    ),
    (
        "Perubahan struktural sendi lutut makin nyata dengan terlihatnya beberapa osteofit "
        "berukuran sedang hingga besar, penyempitan celah sendi yang jelas, dan "
        "dapat disertai sklerosis subkondral serta kemungkinan deformitas tulang "
        "yang mulai tampak. Temuan tersebut sejalan dengan definisi KL Grade 3, "
        "yang dikategorikan sebagai osteoartritis derajat sedang dengan beban "
        "disabilitas yang lebih tinggi [5], [1], [4]."
    ),
    (
        "Osteoartritis derajat berat: penyempitan celah sendi yang sangat jelas "
        "hingga hampir menghilang, osteofit besar dan multipel, sklerosis subkondral "
        "berat, serta deformitas tulang yang nyata (misalnya varus/valgus deformity). "
        "Kriteria ini sesuai dengan KL Grade 4 dan dikaitkan dengan nyeri kronis, "
        "keterbatasan gerak yang signifikan, serta peningkatan kebutuhan intervensi "
        "terapi invasif seperti osteotomi atau artroplasti lutut [5], [1], [4]."
    ),
]

# ---- Konfigurasi Model dari Google Drive
MODEL_DIR = "models"
MODEL_FILENAME = "myKOA90%.h5"  # nama file lokal yang akan disimpan
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# ID dari link Google Drive kamu:
# https://drive.google.com/file/d/1PDVnIHf55rZr5Q2rIxSQE2Ba2tKxOrEb/view?usp=sharing
MODEL_URL = "https://drive.google.com/uc?id=1PDVnIHf55rZr5Q2rIxSQE2Ba2tKxOrEb"

# 🔗 Link Google Drive berisi contoh gambar X-ray (sudah diisi dari kamu)
EXAMPLE_IMAGES_URL = "https://drive.google.com/drive/folders/15Q7tG3CuMR-GsPysYfA5o2wmny36F9yd?usp=sharing"
# 🔗 Link ke User Manual aplikasi myKOA
USER_MANUAL_URL = "https://drive.google.com/drive/folders/1Zfuq-y8K1Cb4YhXp5JSBPb18j96ahwm8?usp=sharing"


def download_model_if_needed():
    """Download model .h5 dari Google Drive jika belum ada di disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        return  # sudah ada, tidak perlu download lagi

    with st.spinner("Mengunduh model KOA (.h5) dari Google Drive, mohon tunggu..."):
        # gdown akan meng-handle link Google Drive (termasuk file besar)
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)


# ---- Load Model
@st.cache_resource
def load_model():
    # Pastikan file model sudah ada (kalau belum, download dulu)
    download_model_if_needed()

    import tensorflow.keras.backend as K

    def focal_loss(gamma=2.0, alpha=0.25):
        def loss(y_true, y_pred):
            y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
            cross_entropy = -y_true * K.log(y_pred)
            weight = alpha * K.pow(1 - y_pred, gamma)
            loss_val = weight * cross_entropy
            return K.mean(K.sum(loss_val, axis=-1))
        return loss

    # ⬇️ load H5 langsung, tidak perlu main-main dengan config manual
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "loss": focal_loss(),
            "focal_loss": focal_loss(),
        },
        compile=False,
    )

    model.compile(optimizer="adam", loss=focal_loss(), metrics=["accuracy"])
    return model



# ---- Fungsi Preprocessing Gambar
def preprocess_image(img_array):
    if img_array.shape[-1] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    img = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    img = np.stack([img]*3, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# ---- Judul & Header
st.markdown(
    "<div class='centered animate-fadein'>"
    "<img src='https://cdn-icons-png.flaticon.com/512/7220/7220514.png' width='110'/>"
    "</div>",
    unsafe_allow_html=True
)
st.markdown("<h1 class='centered animate-fadein'>Yuk periksa tingkat keparahan Osteoarthritis Lututmu dengan myKOA !</h1>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div class='centered animate-fadein' 
         style='margin-top:0.8rem; margin-bottom:0.8rem; display:flex; 
                flex-direction:row; gap:14px; justify-content:center; flex-wrap:wrap;'>

        <a href="{EXAMPLE_IMAGES_URL}" target="_blank" class="gradient-link-btn">
            Lihat Contoh Gambar X-ray
        </a>

        <a href="{USER_MANUAL_URL}" target="_blank" class="gradient-link-btn">
            Buka User Manual myKOA
        </a>

    </div>
    """,
    unsafe_allow_html=True
)



# st.markdown(
#     f"""
    
#     """,
#     unsafe_allow_html=True
# )
st.markdown("<div class='centered animate-fadein'><h3>Unggah gambar X-ray lutut anda!</h3></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='centered animate-fadein'>
    <small>
    Model AI ini akan mengklasifikasikan tingkat keparahan Osteoarthritis (KL Grade 0–4) berdasarkan gambar X-ray.<br/>
    Dikembangkan untuk demo Tugas Akhir dan riset di bidang Data Science dan Inteligent System (DSIS) oleh Yustinus Dwi Adyra.
    </small>
    </div>
    """,
    unsafe_allow_html=True
)

# 🔘 Tombol ke contoh gambar X-ray (posisi strategis di dekat instruksi sebelum uploader)

# ---- Upload Gambar
uploaded_file = st.file_uploader("", type=['jpg','jpeg','png'], key="xray-uploader")

if uploaded_file is not None:
    st.markdown("<div class='card animate-fadein'>", unsafe_allow_html=True)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img_array, caption="Gambar X-ray yang Diunggah")

    loader_placeholder = st.empty()

    with loader_placeholder.container():
        st.markdown(
            """
            <div class="loader-wrapper">
                <div class="loader-pulse">
                    <div class="loader-avatar">
                        🧑‍⚕️
                    </div>
                </div>
                <div class="loader-caption">
                    Analisis citra X-ray berlangsung secara otomatis.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # time.sleep(0.9)
    img_input = preprocess_image(img_array)
    model = load_model()
    preds = model.predict(img_input)
    pred_class = np.argmax(preds, axis=1)[0]
    probas = preds[0]
    # time.sleep(0.3)

    loader_placeholder.empty()

    st.markdown(f"""
        <h3 class='centered animate-fadein'>Prediksi: 
        <span style='color:#2db6f8'>{LABELS[pred_class]}</span></h3>
        <div class='centered' style='margin-bottom:1rem'>
            <div style='font-size:1.05rem; color:#000000; max-width: 620px; text-align:justify; background:#f2fbfd; border-radius:1.2rem; padding:1rem 1.3rem; margin-top:0.9rem; box-shadow:0 2px 12px 0 rgba(62,123,171,0.09)'>
            <b>Keterangan radiografis (berbasis KL Grade):</b><br/>
            {KETERANGAN[pred_class]}
            </div>
        </div>
        <div class='centered'>
            <p style='font-size:13px; color:#000000; max-width:620px; text-align:center;'>
            Deskripsi tingkat keparahan osteoartritis lutut pada aplikasi ini mengacu pada kriteria 
            Kellgren–Lawrence untuk penilaian radiografis osteoartritis [5] dan beberapa tinjauan 
            terkait imaging osteoartritis lutut [1], [4], [7]–[9].
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.progress(float(np.max(probas)), text=f"Keyakinan Model: {int(np.max(probas)*100)}%")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("#### Probabilitas Tiap Kelas")

    fig, ax = plt.subplots()
    bars = ax.bar(LABELS, probas, color=['#2db6f8','#53e2b8','#7cc6fe','#ffe292','#ffaebc'])
    ax.set_ylabel("Probabilitas")
    plt.xticks(rotation=15)
    for bar, prob in zip(bars, probas):
        height = bar.get_height()
        ax.annotate(f"{prob:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    st.pyplot(fig)

    st.markdown(
        "<small>KL Grade = kriteria Kellgren–Lawrence untuk osteoartritis lutut. "
        "0: Normal &bull; 1: Doubtful &bull; 2: Mild &bull; 3: Moderate &bull; 4: Severe.</small>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

else:

    st.markdown("""
     <div class='card animate-fadein'>
        <div class='centered'>
            <img src='https://cdn-icons-png.flaticon.com/512/5900/5900604.png' width='110'/>
        </div>
        <h4 class='centered' style='color:#2db6f8'>Apa itu Knee Osteoarthritis?</h4>
        <p style='text-align:justify; font-size:15px;'>
        <b>Knee Osteoarthritis (OA)</b> adalah penyakit sendi degeneratif yang menyerang tulang rawan lutut dan tulang subkondral, 
        menyebabkan nyeri, kaku, dan keterbatasan fungsi. Secara global, OA lutut merupakan salah satu penyebab utama disabilitas 
        muskuloskeletal pada populasi lanjut usia [1]. Penilaian radiografis OA lutut secara luas menggunakan 
        <b>skala Kellgren–Lawrence (KL Grade)</b> yang mengklasifikasikan tingkat keparahan berdasarkan keberadaan osteofit, 
        penyempitan celah sendi, sklerosis subkondral, dan deformitas tulang [5].
        </p>
        <ul style='font-size:15px;'>
            <li><b>Grade 0</b>: Sendi radiografis normal, tanpa tanda OA.</li>
            <li><b>Grade 1</b>: Osteofit sangat kecil/meragukan, tanpa penyempitan celah sendi yang jelas.</li>
            <li><b>Grade 2</b>: Osteofit definitif dengan kemungkinan penyempitan celah sendi ringan.</li>
            <li><b>Grade 3</b>: Penyempitan celah sendi jelas dengan osteofit multipel dan sklerosis subkondral.</li>
            <li><b>Grade 4</b>: Penyempitan celah sendi berat, osteofit besar, sklerosis berat, dan deformitas tulang.</li>
        </ul>
        <p style='font-size:14px;color:#000000'>
        <i>Model ini menggunakan pendekatan deep learning berbasis U-Net untuk segmentasi dan ResNet50 untuk klasifikasi citra X-ray, 
        sejalan dengan tren pemanfaatan convolutional neural networks untuk analisis citra medis [6], [7]–[12], [14], [15], [17]. 
        Untuk demo, upload gambar X-ray lutut (jpg/png). Semua proses dilakukan secara lokal, tanpa menyimpan gambar Anda.</i>
        </p>
    </div>
    """, unsafe_allow_html=True)

with st.expander("📚 Referensi Ilmiah yang Digunakan di Aplikasi Ini"):
    st.markdown(
        """
        <ol style="font-size:13px; text-align:justify;">
        <li>[1] J. D. Steinmetz et al., “Global, regional, and national burden of osteoarthritis, 1990–2020 and projections to 2050,” <i>The Lancet Rheumatology</i>, 2023.</li>
        <li>[2] D. Mery, “X-ray testing by computer vision,” in <i>Proc. IEEE Conf. Computer Vision and Pattern Recognition Workshops</i>, 2013, pp. 360–367.</li>
        <li>[3] D. Mery and C. Pieringer, <i>Computer Vision for X-Ray Testing</i>, Springer, 2021.</li>
        <li>[4] A. Vashishtha and A. K. Acharya, “An overview of medical imaging techniques for knee osteoarthritis disease,” <i>Biomedical and Pharmacology Journal</i>, vol. 14, no. 2, 2021.</li>
        <li>[5] J. H. Kellgren and J. S. Lawrence, “Radiological assessment of osteo-arthrosis,” <i>Annals of the Rheumatic Diseases</i>, vol. 16, no. 4, pp. 494–502, 1957.</li>
        <li>[6] G. Litjens et al., “Deep learning in medical image analysis,” <i>Medical Image Analysis</i>, vol. 42, pp. 60–88, 2017.</li>
        <li>[7] Y. Wang et al., “A ResNet-based approach for accurate radiographic diagnosis of knee osteoarthritis,” <i>CAAI Transactions on Intelligence Technology</i>, vol. 7, no. 3, pp. 512–521, 2022.</li>
        <li>[8] M. S. Yallappa and G. R. Bharamagoudar, “Classification of knee X-ray images by severity of osteoarthritis using skip connection based ResNet101,” <i>International Journal of Intelligent Engineering &amp; Systems</i>, vol. 16, no. 5, 2023.</li>
        <li>[9] T. Tariq, Z. Suhail, and Z. Nawaz, “Knee osteoarthritis detection and classification using X-rays,” <i>IEEE Access</i>, vol. 11, pp. 48292–48303, 2023.</li>
        <li>[10] H. A. Ahmed and E. A. Mohammed, “Detection and classification of the osteoarthritis in knee joint using transfer learning with CNNs,” <i>Iraqi Journal of Science</i>, 2022.</li>
        <li>[11] S. U. Rehman and V. Gruhn, “A sequential VGG16+CNN based automated approach with adaptive input for efficient detection of knee osteoarthritis stages,” <i>IEEE Access</i>, 2024.</li>
        <li>[12] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in <i>MICCAI 2015</i>, pp. 234–241.</li>
        <li>[13] İ. Altun, S. Altun, and A. Alkan, “LSS-U-Net: Lumbar spinal stenosis semantic segmentation using deep learning,” <i>Multimedia Tools and Applications</i>, vol. 82, no. 26, 2023.</li>
        <li>[14] L. Atika, S. Nurmaini, R. U. Partan, and E. Sukandi, “Image segmentation for mitral regurgitation with CNN based on U-Net, ResNet, V-Net, FractalNet and SegNet,” <i>Big Data and Cognitive Computing</i>, vol. 6, no. 4, p. 141, 2022.</li>
        <li>[15] B. Liang, C. Tang, W. Zhang, M. Xu, and T. Wu, “N-Net: A U-Net architecture with dual encoder for medical image segmentation,” <i>Signal, Image and Video Processing</i>, vol. 17, no. 6, pp. 3073–3081, 2023.</li>
        <li>[16] N.-T. Do, S.-T. Jung, H.-J. Yang, and S.-H. Kim, “Multi-level Seg-UNet model with global and patch-based X-ray images for knee bone tumor detection,” <i>Diagnostics</i>, vol. 11, no. 4, p. 691, 2021.</li>
        <li>[17] S. Gornale and P. Patravali, “Digital Knee X-ray Images,” Mendeley Data, 2020. Available: https://data.mendeley.com/datasets/t9ndx37v5h/1</li>
        <li>[18] M. Cossio, “Augmenting medical imaging: A comprehensive catalogue of 65 techniques for enhanced data analysis,” arXiv:2303.01178, 2023.</li>
        <li>[19] F. Khalvati, E. Kazmierski, dan I. Diamant, “Effect of normalization methods on the predictive performance and reproducibility of radiomic features in brain MRI,” <i>Insights into Imaging</i>, vol. 15, no. 1, 2024.</li>
        <li>[20] A. Younis et al., “Abnormal brain tumors classification using ResNet50 and its comprehensive evaluation,” <i>IEEE Access</i>, 2024.</li>
        <li>[21] S. Aladhadh dan R. Mahum, “Knee osteoarthritis detection using an improved CenterNet dengan pixel-wise voting scheme,” <i>IEEE Access</i>, vol. 11, pp. 22283–22296, 2023.</li>
        <li>[22] S. S. Gornale, P. U. Patravali, dan P. S. Hiremath, “Automatic detection and classification of knee osteoarthritis using Hu’s invariant moments,” <i>Frontiers in Robotics and AI</i>, vol. 7, p. 591827, 2020.</li>
        <li>[23] A. Ghazwan, S. Al-Qazzaz, dan A. A. Abdulmunem, “Radiographic imaging-based joint degradation detection using deep learning,” <i>International Journal of Advanced Technology and Engineering Exploration</i>, vol. 10, no. 108, pp. 1417–1430, 2023.</li>
        </ol>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br/><div class='centered'><small style='color:#8491ae'>© 2025 MyKOA developed by Yustinus Dwi Adyra</small></div>", unsafe_allow_html=True)
