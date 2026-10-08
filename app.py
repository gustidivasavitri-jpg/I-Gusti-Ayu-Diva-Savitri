import html
import json
import re
import unicodedata
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="TELISIK • Baca, cek, pahami",
    page_icon="🔎",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

st.markdown("""
<style>
.stApp {background:#FFF8ED;color:#172554;}
[data-testid="stSidebar"] {background:#EDE9FE;}
h1,h2,h3 {color:#172554;}
.block-container {max-width:1200px;padding-top:2.5rem;}
.hero {
    background:linear-gradient(125deg,#172554,#4935A4);
    padding:32px;border-radius:24px;color:white;
    margin-bottom:18px;
}
.hero h1 {color:white;font-size:clamp(30px,4vw,48px);line-height:1.15;}
.hero p {color:#EDE9FE;font-size:18px;}
.badge {
    display:inline-block;background:#FBBF24;color:#172554;
    padding:6px 12px;border-radius:20px;font-weight:700;
}
.tip {
    background:#EDE9FE;padding:20px;border-radius:16px;
    border-left:5px solid #7C3AED;margin:12px 0;
}
[data-testid="stForm"] {
    background:white;border-radius:18px;padding:22px;
}
div.stButton > button[kind="primary"] {
    background:#6D28D9;border-color:#6D28D9;color:white;
}
</style>
""", unsafe_allow_html=True)


def bersihkan(teks):
    teks = unicodedata.normalize("NFKC", html.unescape(teks))
    return re.sub(r"\s+", " ", teks).strip().lower()


@st.cache_resource
def muat_model():
    return joblib.load(BASE / "model_telisik.joblib")


@st.cache_data
def muat_bank():
    df = pd.read_csv(
        BASE / "bank_kasus_telisik.csv",
        dtype=str,
        keep_default_na=False
    )
    wajib = [
        "id", "topik", "judul", "klaim", "konteks",
        "kesimpulan_sumber", "ringkasan_bukti",
        "penerbit_pemeriksaan", "tanggal_terbit",
        "url_sumber", "pertanyaan_refleksi", "catatan"
    ]
    kurang = set(wajib) - set(df.columns)
    if kurang:
        raise ValueError(f"Kolom CSV belum lengkap: {sorted(kurang)}")
    if df.empty or not df["id"].is_unique:
        raise ValueError("Bank kasus kosong atau ID kasus berulang.")
    return df


bank = muat_bank()
metrik = json.loads(
    (BASE / "metrik.json").read_text(encoding="utf-8")
)

# Indeks ini terpisah dari TF-IDF milik model klasifikasi.
teks_indeks = bank[
    ["judul", "topik", "klaim", "konteks"]
].agg(" ".join, axis=1)

pencari = TfidfVectorizer(ngram_range=(1, 2))
matriks_kasus = pencari.fit_transform(teks_indeks)


def cari_kasus(query):
    vektor = pencari.transform([bersihkan(query)])
    if vektor.nnz == 0:
        return bank.iloc[0:0].copy()
    skor = cosine_similarity(vektor, matriks_kasus).ravel()
    hasil = bank.copy()
    hasil["skor"] = skor
    # Ini hanya penyaring kecocokan kata, bukan batas kebenaran.
    return hasil[hasil["skor"] > 0].sort_values(
        "skor", ascending=False
    ).head(3)


def ilustrasi(topik="Literasi"):
    warna = "#CCFBF1" if topik == "Pertanian" else "#EDE9FE"
    label = html.escape(topik)
    # Ilustrasi SVG dekoratif, bukan foto atau bukti kasus.
    st.markdown(f"""
    <svg viewBox="0 0 500 210" width="100%"
         role="img" aria-label="Ilustrasi pemeriksaan informasi">
      <rect width="500" height="210" rx="24" fill="{warna}"/>
      <circle cx="400" cy="45" r="45" fill="#FBBF24" opacity=".7"/>
      <rect x="90" y="28" width="210" height="154" rx="14"
            fill="white" stroke="#172554" stroke-width="3"/>
      <rect x="111" y="50" width="76" height="37" rx="7"
            fill="#14B8A6"/>
      <path d="M203 56H277 M203 75H265 M112 107H273
               M112 128H245 M112 149H219"
            stroke="#A5B4FC" stroke-width="8" stroke-linecap="round"/>
      <circle cx="319" cy="105" r="44" fill="#FBBF24"
              fill-opacity=".5" stroke="#172554" stroke-width="10"/>
      <path d="M350 139L391 180" stroke="#172554"
            stroke-width="17" stroke-linecap="round"/>
      <text x="20" y="195" fill="#172554" font-size="14"
            font-family="sans-serif">ILUSTRASI • {label}</text>
    </svg>
    """, unsafe_allow_html=True)


def kartu(row, gambar=True):
    with st.container(border=True):
        if gambar:
            ilustrasi(row["topik"])
        st.caption(f'{row["topik"]} • Arsip {row["tanggal_terbit"]}')
        st.subheader(row["judul"])
        st.write("**Klaim yang diperiksa:**")
        st.write(row["klaim"])
        with st.expander("Buka konteks dan hasil pemeriksaan"):
            st.write(row["konteks"])
            st.write(
                "**Kesimpulan sumber untuk kasus ini:**",
                row["kesimpulan_sumber"]
            )
            st.write(row["ringkasan_bukti"])
            st.caption(row["catatan"])
            st.write("**Pertanyaan refleksi:**")
            st.write(row["pertanyaan_refleksi"])
        if row["url_sumber"].startswith("https://"):
            st.link_button("Baca pemeriksaan sumber", row["url_sumber"])
        st.caption(row["penerbit_pemeriksaan"])


st.sidebar.title("🔎 TELISIK")
st.sidebar.caption("Baca. Cek. Pahami.")
menu = st.sidebar.radio(
    "Jelajahi",
    ["Beranda", "Jelajah Fakta", "Coba AI",
     "Belajar 2 Menit", "Tentang Model"]
)
st.sidebar.divider()
st.sidebar.write(f"📚 {len(bank)} kasus dalam koleksi")
st.sidebar.caption(
    "Koleksi diperbarui manual. Prediksi AI dan hasil "
    "pemeriksaan sumber ditampilkan secara terpisah."
)

if menu == "Beranda":
    kiri, kanan = st.columns([1.3, 1])
    with kiri:
        st.markdown("""
        <div class="hero">
          <span class="badge">RUANG BELAJAR LITERASI KRITIS</span>
          <h1>Jangan buru-buru percaya.<br>Telisik dulu ceritanya.</h1>
          <p>Baca kasus, uji penilaianmu, dan buka sumber
          sebelum mengambil kesimpulan.</p>
        </div>
        """, unsafe_allow_html=True)
    with kanan:
        ilustrasi()
        st.caption("Ilustrasi dekoratif: menelusuri informasi.")

    a, b, c = st.columns(3)
    a.info("**01 · BACA**\n\nKenali klaim dan konteksnya.")
    b.info("**02 · TELISIK**\n\nPeriksa sumber serta buktinya.")
    c.info("**03 · PIKIRKAN ULANG**\n\nTinjau kembali kesimpulanmu.")

    st.header("Kasus pilihan")
    st.caption(
        "Arsip pembelajaran, bukan berita terbaru. "
        "Tanggal pada kartu adalah tanggal publikasi sumber."
    )
    kolom = st.columns(2)
    for i, (_, row) in enumerate(bank.head(4).iterrows()):
        with kolom[i % 2]:
            kartu(row)

elif menu == "Jelajah Fakta":
    st.title("Temukan rujukan, baca konteksnya")
    st.write(
        "Pencarian mencocokkan kata dalam koleksi TELISIK. "
        "Hasilnya belum tentu membahas klaim yang persis sama."
    )
    query = st.text_input(
        "Apa yang ingin kamu telisik?",
        placeholder="Contoh: beras plastik atau bantuan pupuk"
    )
    topik = st.selectbox(
        "Topik", ["Semua"] + sorted(bank["topik"].unique().tolist())
    )

    # Batasi koleksi sesuai topik sebelum memilih hasil teratas.
    if query.strip():
        v = pencari.transform([bersihkan(query)])
        hasil = bank.copy()
        hasil["skor"] = cosine_similarity(v, matriks_kasus).ravel()
        hasil = hasil[hasil["skor"] > 0]
        hasil = hasil.sort_values("skor", ascending=False)
    else:
        hasil = bank.copy()

    if topik != "Semua":
        hasil = hasil[hasil["topik"] == topik]

    if hasil.empty:
        st.info(
            "Belum ditemukan rujukan dengan kecocokan kata "
            "dalam koleksi ini. Ini bukan berarti klaim benar atau salah."
        )
    else:
        st.caption(f"{len(hasil)} kasus ditemukan.")
        for _, row in hasil.head(10).iterrows():
            kartu(row)

elif menu == "Coba AI":
    st.title("Penilaianmu dan prediksi AI")
    st.write(
        "Gunakan artikel berbahasa Indonesia secara utuh. "
        "Model belum divalidasi untuk klaim pendek."
    )
    st.caption(
        "Alasanmu disimpan dalam sesi untuk refleksi; "
        "aplikasi belum menilai kualitas alasan secara otomatis."
    )

    with st.form("form_ai"):
        teks = st.text_area("Tempel artikel", height=200, max_chars=30000)
        penilaian = st.radio(
            "Menurutmu artikel ini…",
            ["Cenderung Hoax", "Cenderung Valid", "Belum yakin"],
            index=None
        )
        alasan = st.text_area("Apa alasanmu?", max_chars=2000)
        kirim = st.form_submit_button("Telisik bersama", type="primary")

    if kirim:
        st.session_state.pop("hasil_ai", None)
        if not teks.strip() or penilaian is None or not alasan.strip():
            st.warning("Isi artikel, pilihan penilaian, dan alasanmu.")
        else:
            model = muat_model()
            bersih = bersihkan(teks)
            fitur = model.named_steps["tfidf"].transform([bersih])
            if fitur.nnz == 0:
                st.warning(
                    "Tidak ada fitur teks yang dikenali model. "
                    "Prediksi tidak ditampilkan."
                )
            else:
                pred = str(model.predict([bersih])[0])
                st.session_state["hasil_ai"] = {
                    "teks": teks, "penilaian": penilaian,
                    "alasan": alasan, "prediksi": pred
                }

    if "hasil_ai" in st.session_state:
        h = st.session_state["hasil_ai"]
        st.caption("Hasil untuk kiriman terakhir. Kirim ulang jika teks diubah.")
        with st.expander("Lihat teks yang dianalisis"):
            st.write(h["teks"])

        a, b = st.columns(2)
        a.info(f"**Penilaianmu**\n\n{h['penilaian']}")
        b.info(f"**Prediksi SVM**\n\nCenderung {h['prediksi']}")
        st.write("**Alasanmu:**", h["alasan"])

        if h["penilaian"] == "Belum yakin":
            st.write("Kamu belum menyimpulkan. Lanjutkan dengan memeriksa sumber.")
        elif h["penilaian"] == f"Cenderung {h['prediksi']}":
            st.write(
                "Penilaianmu sejalan dengan model. "
                "Kesamaan ini belum membuktikan kebenaran."
            )
        else:
            st.write(
                "Penilaianmu berbeda dari model. "
                "Perbedaan ini tidak otomatis berarti kamu salah."
            )

        st.warning(
            "SVM membaca pola teks, bukan memverifikasi fakta. "
            "Prediksi dapat keliru."
        )
        st.subheader("Telusuri rujukan dalam koleksi")
        st.caption(
            "Kartu berikut hanya cocok secara kata. Periksa apakah "
            "objek, kejadian, waktu, dan klaimnya benar-benar sama."
        )
        terkait = cari_kasus(h["teks"])
        if terkait.empty:
            st.info("Belum ditemukan rujukan yang cocok secara kata.")
        else:
            for _, row in terkait.iterrows():
                kartu(row, gambar=False)

        st.markdown("""
        **Sebelum menyimpulkan:**
        - Siapa sumber asli artikel?
        - Apakah tanggal dan konteksnya sesuai?
        - Apakah bukti mendukung klaim?
        - Apakah artikel mengutip klaim untuk membantahnya?
        """)

elif menu == "Belajar 2 Menit":
    st.title("Sedikit membaca, lebih teliti menilai")
    materi = [
        ("Ramai dibagikan ≠ terbukti benar",
         "Jumlah pembagian menunjukkan jangkauan informasi. "
         "Untuk menilai kebenaran, cari sumber asli dan bukti."),
        ("Bedakan kutipan dengan kesimpulan",
         "Artikel bisa mengutip hoaks lalu membantahnya. "
         "Baca sampai akhir sebelum menilai isi artikel."),
        ("AI juga bisa keliru",
         "Prediksi model bergantung pada pola data latih. "
         "Gunakan bukti yang dapat diperiksa untuk menyimpulkan.")
    ]
    for judul, isi in materi:
        st.markdown(
            f'<div class="tip"><h3>{judul}</h3><p>{isi}</p></div>',
            unsafe_allow_html=True
        )

    st.subheader("Coba satu situasi")
    st.write(
        "Sebuah pesan menyertakan logo instansi dan meminta "
        "data pribadi melalui tautan. Apa langkah paling tepat?"
    )
    with st.form("kuis"):
        jawaban = st.radio(
            "Pilih tindakanmu",
            ["Langsung isi karena ada logo.",
             "Periksa pengumuman dan alamat situs resmi instansi.",
             "Bagikan dahulu agar teman ikut mendaftar."],
            index=None
        )
        cek = st.form_submit_button("Lihat pembahasan")
    if cek:
        if jawaban is None:
            st.warning("Pilih jawaban terlebih dahulu.")
        else:
            tepat = jawaban == (
                "Periksa pengumuman dan alamat situs resmi instansi."
            )
            st.info("Jawaban tepat." if tepat else "Coba tinjau kembali pilihanmu.")
            st.write(
                "Logo mudah disalin. Buka kanal resmi instansi "
                "secara mandiri dan cocokkan pengumumannya."
            )
            st.caption(
                "Situasi ini adalah ilustrasi pembelajaran. "
                "Satu jawaban tidak mengukur keseluruhan literasimu."
            )

else:
    st.title("Kenali kemampuan dan batas TELISIK")
    a, b, c = st.columns(3)
    a.metric("Akurasi pada data uji", f"{metrik['akurasi']:.2%}")
    b.metric("F1-macro", f"{metrik['f1_macro']:.4f}")
    c.metric("Artikel uji", metrik["jumlah_uji"])

    st.write(
        f"Model TF-IDF + SVM dilatih menggunakan "
        f"{metrik['jumlah_latih']} artikel."
    )
    st.dataframe(pd.DataFrame(
        metrik["confusion_matrix"],
        index=["Rujukan Hoax", "Rujukan Valid"],
        columns=["Prediksi Hoax", "Prediksi Valid"]
    ), use_container_width=True)

    st.write(
        "Eksperimen memakai 550 dari 600 artikel. Sebanyak 50 "
        "artikel dipisahkan sementara karena kandidat konflik label. "
        "Kelompok kemiripan dipisahkan antara latihan dan pengujian."
    )
    st.write(
        "Metrik berasal dari satu pembagian uji dataset lama; "
        "bukan ukuran akurasi untuk semua informasi baru atau "
        "bukti peningkatan literasi pengguna."
    )
    st.markdown(
        "**Dataset model:** Rahutomo, Yanuar, dan Asmara (2018), "
        "[Indonesian Hoax News Detection Dataset]"
        "(https://data.mendeley.com/datasets/p3hfgr5j3m/1), CC BY 4.0."
    )
    st.info(
        f"Bank rujukan berisi {len(bank)} kasus. "
        "Bank ini tidak digunakan untuk melatih ulang SVM. "
        "Pencarian memakai TF-IDF dan cosine similarity, "
        "bukan chatbot atau verifikasi fakta otomatis."
    )

st.divider()
st.caption("TELISIK • Baca lebih teliti. Nilai dengan alasan.")
