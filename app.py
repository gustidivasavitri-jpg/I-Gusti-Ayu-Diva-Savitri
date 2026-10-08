
import json
import re
import html
import unicodedata
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="TELISIK | Literasi Kritis",
    page_icon="🔎",
    layout="centered"
)

BASE = Path(__file__).resolve().parent

@st.cache_resource
def muat_model():
    return joblib.load(BASE / "model_telisik.joblib")

def bersihkan(teks):
    teks = html.unescape(teks)
    teks = unicodedata.normalize("NFKC", teks)
    return re.sub(r"\s+", " ", teks).strip().lower()

model = muat_model()
metrik = json.loads(
    (BASE / "metrik.json").read_text(encoding="utf-8")
)

st.title("🔎 TELISIK")
st.caption("Baca lebih teliti. Nilai dengan alasan.")

menu = st.sidebar.radio(
    "Mulai dari sini",
    ["Latihan Literasi", "Coba Model AI", "Data & Evaluasi"]
)
st.sidebar.caption(
    "Prototipe pembelajaran berbasis TF-IDF dan SVM."
)

if menu == "Latihan Literasi":
    st.header("Latih cara menilai informasi")
    st.write(
        "Kasus berikut merupakan ilustrasi pembelajaran. "
        "Pilih tindakan yang paling tepat sebelum membaca pembahasan."
    )

    soal = [
        {
            "kasus": (
                "Sebuah pesan sudah dibagikan ribuan kali, "
                "tetapi tidak menyertakan sumber."
            ),
            "opsi": [
                "Langsung percaya karena banyak yang membagikan.",
                "Cari sumber asli dan bukti pendukungnya.",
                "Langsung menyatakan pesan itu palsu."
            ],
            "benar": 1,
            "bahas": (
                "Jumlah pembagian tidak membuktikan kebenaran. "
                "Cari sumber asli, tanggal, dan bukti sebelum menyimpulkan."
            )
        },
        {
            "kasus": (
                "Artikel memuat sebuah klaim sensasional di awal, "
                "lalu membantah klaim tersebut pada bagian berikutnya."
            ),
            "opsi": [
                "Nilai artikel dari kalimat pertamanya saja.",
                "Anggap semua kalimat dalam artikel memiliki maksud sama.",
                "Baca keseluruhan dan bedakan kutipan dengan kesimpulan."
            ],
            "benar": 2,
            "bahas": (
                "Artikel bantahan dapat mengutip informasi salah. "
                "Konteks keseluruhan diperlukan untuk memahami sikap penulis."
            )
        },
        {
            "kasus": (
                "AI memprediksi sebuah artikel sebagai Valid, "
                "tetapi sumber artikel belum kamu periksa."
            ),
            "opsi": [
                "Tetap periksa sumber dan bukti artikelnya.",
                "Bagikan karena AI pasti benar.",
                "Anggap prediksi AI sebagai bukti pendukung."
            ],
            "benar": 0,
            "bahas": (
                "Model dapat salah. Prediksi klasifikasi merupakan "
                "hasil perhitungan pola teks, bukan verifikasi fakta."
            )
        }
    ]

    with st.form("latihan"):
        jawaban = []
        for i, s in enumerate(soal):
            st.subheader(f"Kasus {i + 1}")
            st.write(s["kasus"])
            jawaban.append(st.radio(
                "Apa tindakanmu?",
                s["opsi"],
                index=None,
                key=f"soal_{i}"
            ))
        kirim = st.form_submit_button("Lihat pembahasan")

    if kirim:
        if any(j is None for j in jawaban):
            st.warning("Jawab ketiga kasus terlebih dahulu.")
        else:
            skor = sum(
                j == s["opsi"][s["benar"]]
                for j, s in zip(jawaban, soal)
            )
            st.success(f"Jawaban tepat: {skor} dari {len(soal)}")
            for i, s in enumerate(soal):
                st.markdown(f"**Kasus {i + 1}**")
                st.write(s["bahas"])
            st.caption(
                "Skor ini hanya untuk tiga latihan tersebut, "
                "bukan ukuran menyeluruh tingkat literasi."
            )

elif menu == "Coba Model AI":
    st.header("Bandingkan penilaianmu dengan AI")
    st.info(
        "Gunakan teks artikel berbahasa Indonesia secara utuh. "
        "Model ini belum divalidasi untuk klaim pendek atau semua topik."
    )

    with st.form("prediksi"):
        teks = st.text_area(
            "Tempel teks artikel",
            height=230,
            max_chars=30000
        )
        penilaian = st.radio(
            "Penilaian awalmu",
            ["Cenderung Hoax", "Cenderung Valid", "Belum yakin"]
        )
        alasan = st.text_area("Apa alasanmu?", max_chars=2000)
        proses = st.form_submit_button("Bandingkan dengan AI")

    if proses:
        if not teks.strip() or not alasan.strip():
            st.warning("Isi teks artikel dan alasan penilaianmu.")
        else:
            teks_bersih = bersihkan(teks)
            fitur = model.named_steps["tfidf"].transform([teks_bersih])

            if fitur.nnz == 0:
                st.warning(
                    "Tidak ditemukan fitur yang dikenali model. "
                    "Prediksi tidak ditampilkan."
                )
            else:
                prediksi = model.predict([teks_bersih])[0]
                st.subheader(f"Prediksi model: {prediksi}")
                st.write(f"**Penilaian awalmu:** {penilaian}")
                st.write("**Alasanmu:**")
                st.write(alasan)
                st.warning(
                    "Prediksi ini bukan hasil pemeriksaan fakta. "
                    "Aplikasi tidak menelusuri sumber atau internet."
                )
                st.markdown(
                    "**Lanjutkan pemeriksaan:**\n"
                    "- Siapa sumber asli informasinya?\n"
                    "- Kapan dan dalam konteks apa informasi diterbitkan?\n"
                    "- Bukti apa yang mendukung kesimpulannya?\n"
                    "- Apakah sumber independen mendukung informasi tersebut?"
                )
                st.caption(
                    "Kesamaan jawabanmu dengan AI tidak otomatis "
                    "berarti jawabanmu benar."
                )

else:
    st.header("Kenali data dan batas model")
    a, b = st.columns(2)
    a.metric("Akurasi uji", f"{metrik['akurasi']:.2%}")
    b.metric("F1-macro", f"{metrik['f1_macro']:.4f}")

    st.write(
        f"Data latih: {metrik['jumlah_latih']} berita. "
        f"Data uji: {metrik['jumlah_uji']} berita."
    )

    st.dataframe(pd.DataFrame(
        metrik["confusion_matrix"],
        index=["Rujukan Hoax", "Rujukan Valid"],
        columns=["Prediksi Hoax", "Prediksi Valid"]
    ))

    st.write(
        "Eksperimen menggunakan 550 dari 600 berita. "
        "Sebanyak 50 berita dipisahkan sementara karena menjadi "
        "kandidat konflik label pada teks sangat mirip. "
        "Kelompok kemiripan dipisahkan antara data latih dan uji."
    )
    st.write(
        "Hasil ini berlaku pada satu pembagian uji dataset berita lama. "
        "Belum merupakan evaluasi antartopik, pemeriksaan fakta terkini, "
        "atau bukti peningkatan literasi pengguna."
    )
    st.markdown(
        "**Sumber dataset:** Faisal Rahutomo, Inggrid Yanuar, "
        "dan Rosa Andrie Asmara (2018), "
        "[Indonesian Hoax News Detection Dataset]"
        "(https://data.mendeley.com/datasets/p3hfgr5j3m/1). "
        "Lisensi CC BY 4.0. Teks dinormalisasi dan "
        "kandidat konflik dipisahkan untuk eksperimen."
    )

st.divider()
st.caption("TELISIK — Belajar menilai informasi bersama AI.")
