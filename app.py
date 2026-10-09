import base64
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import joblib
import pandas as pd
import streamlit as st
from core import clean, search_cases, fetch_feed, FEEDS

BASE=Path(__file__).resolve().parent
st.set_page_config(page_title='TELISIK — Berani cek, bijak percaya.',page_icon='✳',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--ink:#202139;--paper:#FAF8F4;--orange:#FF743D;--lilac:#C6B8FF}
.stApp{background:var(--paper);color:var(--ink);font-family:'DM Sans',sans-serif}
h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.04em;color:var(--ink)}
.block-container{max-width:1280px;padding-top:1.4rem;padding-bottom:3rem}
header[data-testid="stHeader"]{background:#FAF8F4DD}
.brand{display:flex;align-items:center;gap:12px;font-size:28px;font-weight:800;letter-spacing:-1px}
.brand b{background:#FF743D;padding:0 10px;border-radius:12px;font-size:30px}
.brand small{font-size:10px;letter-spacing:1.5px;font-weight:600;border-left:1px solid #ccc;padding-left:15px;margin-left:6px;line-height:1.5}
.eyebrow{font-size:11px;letter-spacing:2px;font-weight:700;text-transform:uppercase;margin-bottom:14px}
.hero{padding:24px 0 16px}.hero h1{font-size:clamp(44px,5.7vw,75px)!important;line-height:1.04!important;margin:12px 0 18px!important;font-weight:700!important;letter-spacing:-4px}
.hero h1 em{font-style:normal;color:#EE6630}.hero p{max-width:440px;color:#626273;font-size:17px;line-height:1.7}
.pill{display:inline-block;border:1px solid #D9D3CA;border-radius:30px;padding:7px 12px;font-size:11px;font-weight:600;margin-right:6px}
.ribbon{background:#202139;color:white;border-radius:18px;padding:18px 24px;display:flex;gap:24px;align-items:center;justify-content:space-between;margin:22px 0 28px;font-size:13px}
.ribbon b{color:#C6B8FF;font-size:18px}.section-label{color:#797580;font-size:11px;letter-spacing:2px;margin-top:24px}
.art{width:100%;border-radius:20px;display:block}.tag{display:inline-block;background:#EEE8FF;color:#514273;border-radius:6px;font-size:10px;font-weight:700;padding:4px 8px;margin:12px 0 5px;letter-spacing:1px;text-transform:uppercase}
.case-title{font-family:'Space Grotesk',sans-serif;font-size:23px;line-height:1.18;font-weight:600;letter-spacing:-.6px;margin:6px 0 10px;min-height:55px}.case-meta{font-size:11px;color:#777585;margin-bottom:12px}
.feature{border-radius:20px;background:#EEE8FF;padding:23px;min-height:200px}.feature.orange{background:#FFE6D6}.feature.green{background:#E2F2E6}.feature span{font-size:11px;font-weight:700;letter-spacing:2px}.feature h3{font-size:25px;margin-top:18px}.feature p{color:#565467;font-size:14px}
.scorebox{padding:20px 24px;background:#202139;color:white;border-radius:18px;margin-bottom:18px}.scorebox b{font-size:30px;color:#C6B8FF}.step{font-size:11px;letter-spacing:1.6px;font-weight:700;color:#66569B;margin:12px 0}
.quote{font-family:'Space Grotesk',sans-serif;font-size:25px;line-height:1.45;background:#EEE8FF;padding:28px;border-radius:20px;margin:16px 0}
[data-testid="stVerticalBlockBorderWrapper"]>div{border-radius:20px!important}
[data-testid="stForm"]{background:white;border-radius:20px;padding:22px}
[data-testid="stMetric"]{background:#EEE8FF;padding:16px 20px;border-radius:16px}
.stButton button,.stLinkButton a,.stDownloadButton button{border-radius:12px!important;font-weight:600!important}
.stButton button[kind="primary"]{background:#202139!important;color:white!important;border:1px solid #202139!important}
.stButton button[kind="primary"]:hover{background:#514273!important;border-color:#514273!important}
[data-testid="stRadio"]>div[role="radiogroup"]{gap:10px}
[data-testid="stRadio"] label{background:#F0EDE7;padding:7px 12px;border-radius:10px}
.footer{border-top:1px solid #DDD8D0;margin-top:45px;padding-top:22px;color:#7D7885;font-size:12px;display:flex;justify-content:space-between;gap:12px}
@media(max-width:700px){.hero h1{letter-spacing:-2px}.ribbon{flex-wrap:wrap;gap:12px}.brand small{display:none}.case-title{min-height:0}.block-container{padding:1.2rem}.footer{flex-wrap:wrap}}
@media(prefers-reduced-motion:no-preference){.art{transition:transform .25s ease}.art:hover{transform:translateY(-3px)}}
</style>''',unsafe_allow_html=True)

@st.cache_data
def load_cases():
    cases=json.loads((BASE/'kasus.json').read_text(encoding='utf-8'))
    if not cases or len({c['id'] for c in cases})!=len(cases): raise ValueError('Bank kasus tidak valid.')
    return cases
@st.cache_resource
def load_model(): return joblib.load(BASE/'model_telisik.joblib')
@st.cache_data(ttl=900,show_spinner=False)
def news(channel): return fetch_feed(channel)

def esc(s): return html.escape(str(s))
def art(name):
    pilihan = [
        BASE / "assets" / f"{name}.svg",
        BASE / f"{name}.svg",
    ]

    path = next((p for p in pilihan if p.is_file()), None)

    if path is None:
        st.warning(f"Gambar {name}.svg belum ditemukan.")
        return

    encoded = base64.b64encode(path.read_bytes()).decode()
    st.markdown(
        f'<img class="art" alt="Ilustrasi editorial" '
        f'src="data:image/svg+xml;base64,{encoded}">',
        unsafe_allow_html=True
    )
def goto(page,case=None):
    st.session_state['_pending_nav']=page
    if case: st.session_state['selected_case']=case
    st.rerun()
def timestamp(value):
    try:return datetime.fromisoformat(value).astimezone(ZoneInfo('Asia/Jakarta')).strftime('%d %b %Y · %H:%M WIB')
    except (ValueError,TypeError):return value or 'Tanggal tidak tersedia'
def eyebrow(s): st.markdown(f'<div class="eyebrow">{esc(s)}</div>',unsafe_allow_html=True)

CASES=load_cases()
METRICS=json.loads((BASE/'metrik.json').read_text())
for key,default in [('nav','Beranda'),('saved',[]),('journal',{}),('quiz_answers',{}),('case_steps',{}),('ai_history',[])]:
    if key not in st.session_state:st.session_state[key]=default

st.markdown('<div class="brand"><b>✳</b> TELISIK <small>BERANI CEK.<br>BIJAK PERCAYA.</small></div>',unsafe_allow_html=True)
st.write('')
if '_pending_nav' in st.session_state:
    st.session_state['nav']=st.session_state.pop('_pending_nav')
page=st.radio('Navigasi',['Beranda','Radar Informasi','Jelajah Kasus','Lab AI','Misi Literasi','Progres'],horizontal=True,key='nav',label_visibility='collapsed')
st.divider()

def card(c,prefix):
    with st.container(border=True):
        art(c['art'])
        st.markdown(f'<span class="tag">{esc(c["topic"])}</span><div class="case-title">{esc(c["title"])}</div><div class="case-meta">ARSIP · {esc(c["date"])} · {esc(c["source"])}</div>',unsafe_allow_html=True)
        if st.button('Telisik kasus ↗',key=prefix+c['id'],use_container_width=True):goto('Jelajah Kasus',c['id'])

def reference(c):
    st.caption(f'{c["source"]} · terbit {c["date"]} · koleksi ditinjau {c["reviewed"]}')
    st.link_button('Buka pemeriksaan asli ↗',c['url'],use_container_width=True)

if page=='Beranda':
    left,right=st.columns([1.05,1],gap='large')
    with left:
        st.markdown('''<div class="hero"><div class="eyebrow">BUKAN SEKADAR SCROLL. SAATNYA TELISIK.</div><h1>Informasi cepat.<br>Pikiran tetap<br><em>cermat.</em></h1><p>Temukan konteks di balik klaim. Latih penilaianmu, bandingkan dengan AI, lalu buka buktinya.</p></div>''',unsafe_allow_html=True)
        a,b=st.columns(2)
        if a.button('Mulai misi literasi →',type='primary',use_container_width=True):goto('Misi Literasi')
        if b.button('Jelajahi kasus ↗',use_container_width=True):goto('Jelajah Kasus')
        st.markdown('<p style="margin-top:20px"><span class="pill">Tanpa akun</span><span class="pill">Sumber dapat dibuka</span><span class="pill">Belajar bertahap</span></p>',unsafe_allow_html=True)
    with right:
        st.write('');art('hero');st.caption('Ilustrasi editorial TELISIK • bukan foto bukti')
    st.markdown(f'<div class="ribbon"><span><b>{len(CASES):02d}</b> kasus terkurasi</span><span><b>08</b> misi penalaran</span><span><b>RSS</b> berita dari sumber</span><span><b>AI + kamu</b> ruang refleksi</span></div>',unsafe_allow_html=True)
    eyebrow('PILIHAN EDITOR • KOLEKSI 8 OKTOBER 2026')
    st.header('Kelihatannya meyakinkan. Sudah diperiksa?')
    cols=st.columns(3)
    for col,c in zip(cols,CASES[:3]):
        with col:card(c,'home')
    st.caption('Arsip pemeriksaan, bukan umpan otomatis. Setiap kesimpulan mengikuti waktu dan objek yang diperiksa sumber.')
    st.write('');st.header('Satu kebiasaan kecil. Tiga cara berlatih.')
    for col,cl,n,t,d in zip(st.columns(3),['','orange','green'],['01','02','03'],['Buka konteks','Uji pikiranmu','Bandingkan dengan AI'],['Kasus nyata, bukti tertaut, dan refleksi sebelum–sesudah.','Delapan skenario untuk mengenali cara informasi menyesatkan.','Lihat prediksi, pola kata, dan keterbatasan model secara terbuka.']):
        col.markdown(f'<div class="feature {cl}"><span>{n} / RUANG BELAJAR</span><h3>{t}</h3><p>{d}</p></div>',unsafe_allow_html=True)
    st.write('')
    if st.button('Lihat pembaruan berita di Radar Informasi ↗'):goto('Radar Informasi')

elif page=='Radar Informasi':
    eyebrow('SUMBER BERGERAK • KAMU TETAP KRITIS')
    st.title('Apa yang sedang diberitakan?')
    st.write('Judul dari RSS resmi ANTARA. Ini berita untuk dibaca kritis, bukan hasil pemeriksaan fakta oleh TELISIK.')
    a,b=st.columns([3,1])
    channel=a.selectbox('Pilih kanal',list(FEEDS))
    refresh=b.button('↻ Ambil pembaruan',use_container_width=True)
    if refresh:news.clear()
    cache_key='feed_last_'+channel
    try:
        with st.spinner('Mengambil judul dari sumber…'):feed=news(channel)
        st.session_state[cache_key]=feed
        live_ok=True
    except Exception:
        feed=st.session_state.get(cache_key)
        live_ok=False
        st.warning('Sumber belum dapat diakses. Coba lagi nanti atau buka kanal resmi. Arsip kasus tidak menggantikan berita terbaru.')
    if feed:
        st.caption(('Berhasil diambil' if live_ok else 'Salinan terakhir dalam sesi')+' · '+timestamp(feed['fetched_at'])+' · cache maksimal 15 menit saat halaman digunakan.')
        if feed['items'][0]['timestamp'] and (datetime.now().timestamp()-feed['items'][0]['timestamp'])>172800:
            st.info('Berita teratas dalam umpan berusia lebih dari dua hari. Waktu pengambilan bukan tanggal terbit berita.')
        keyword=st.text_input('Saring judul',placeholder='Ketik topik yang ingin kamu baca')
        items=[i for i in feed['items'] if keyword.lower() in i['title'].lower()]
        if not items:st.info('Tidak ada judul yang cocok dalam umpan ini.')
        for i,item in enumerate(items):
            with st.container(border=True):
                a,b=st.columns([5,1])
                a.caption('ANTARA • '+timestamp(item['date']))
                a.subheader(item['title'])
                b.link_button('Baca ↗',item['url'],use_container_width=True)
                with st.expander('Pakai berita ini untuk latihan membaca kritis'):
                    st.write('Buka artikel lengkap. Apa klaim utamanya? Siapa yang dikutip? Bukti apa yang masih perlu diperiksa?')
                    st.text_area('Catatan bacaanmu',key='news_note_'+item['url'])
    st.link_button('Buka ANTARA langsung ↗','https://www.antaranews.com/')
    st.divider();st.subheader('Mencari pemeriksaan fakta terbaru?')
    a,b=st.columns(2)
    a.link_button('TurnBackHoax ↗','https://turnbackhoax.id/articles',use_container_width=True)
    b.link_button('CekFakta ↗','https://cekfakta.com/',use_container_width=True)

elif page=='Jelajah Kasus':
    eyebrow('BACA KLAIM • BUKA BUKTI • TINJAU ULANG')
    st.title('Kamu yang menyelidiki.')
    st.caption('Koleksi awal berfokus pada contoh klaim bermasalah. Ini bukan sampel seimbang atau dataset untuk melatih klasifikasi.')
    selected=st.session_state.get('selected_case')
    if selected:
        c=next((c for c in CASES if c['id']==selected),None)
        if not c:
            st.session_state.pop('selected_case',None);st.rerun()
        if st.button('← Kembali ke semua kasus'):
            st.session_state.pop('selected_case',None);st.rerun()
        l,r=st.columns([1.35,1],gap='large')
        with r:art(c['art']);reference(c)
        with l:
            st.caption(c['topic']+' · '+c['date']);st.title(c['title'])
            st.markdown(f'<div class="quote">{esc(c["claim"])}</div>',unsafe_allow_html=True)
            saved=c['id'] in st.session_state.saved
            if st.button('Hapus dari simpanan' if saved else 'Simpan kasus ☆'):
                if saved:st.session_state.saved.remove(c['id'])
                else:st.session_state.saved.append(c['id'])
                st.rerun()
        record=st.session_state.case_steps.get(c['id'])
        st.progress(1.0 if c['id'] in st.session_state.journal else .66 if record else .33)
        st.markdown('<div class="step">01 / PENILAIAN AWAL</div>',unsafe_allow_html=True)
        with st.form('before_'+c['id']):
            choice=st.radio('Sebelum membuka bukti, kamu…',['Cenderung percaya','Cenderung meragukan','Belum cukup informasi'],index=None)
            reason=st.text_area('Apa yang membuatmu berpikir demikian?')
            confidence=st.slider('Seberapa yakin kamu terhadap penilaianmu?',0,100,50,format='%d%%')
            submit=st.form_submit_button('Simpan penilaian & buka bukti →',type='primary')
        if submit:
            if choice is None or not reason.strip():st.warning('Pilih penilaian dan tulis alasan singkat terlebih dahulu.')
            else:
                st.session_state.case_steps[c['id']]={'choice':choice,'reason':reason,'confidence':confidence}
                st.session_state.journal.pop(c['id'],None)
                st.rerun()
        if record:
            st.markdown('<div class="step">02 / JEJAK PEMERIKSAAN</div>',unsafe_allow_html=True)
            st.info(c['evidence'])
            st.write('**Kesimpulan pemeriksa untuk kasus ini:** '+c['verdict'])
            st.write('**Kebiasaan yang dilatih:** '+c['lesson'])
            reference(c)
            st.caption('Ringkasan ditulis ulang untuk pembelajaran. Buka sumber asli untuk bukti lengkap dan koreksi. Tidak berlaku otomatis pada unggahan lain.')
            st.markdown('<div class="step">03 / PIKIRKAN ULANG</div>',unsafe_allow_html=True)
            with st.form('after_'+c['id']):
                conclusion=st.radio('Setelah membaca bukti…',['Penilaian saya berubah','Penilaian saya tetap','Saya masih perlu bukti lain'],index=None)
                reflection=st.text_area('Bukti apa yang paling memengaruhi kesimpulanmu?')
                finish=st.form_submit_button('Simpan refleksi ✓',type='primary')
            if finish:
                if conclusion is None or not reflection.strip():st.warning('Lengkapi kesimpulan dan refleksimu.')
                else:
                    st.session_state.journal[c['id']]={**record,'title':c['title'],'after':conclusion,'reflection':reflection,'source':c['url']}
                    st.rerun()
            if c['id'] in st.session_state.journal:
                st.success('Refleksimu tersimpan dalam sesi. Unduh catatan di menu Progres.')
    else:
        a,b=st.columns([3,1]);q=a.text_input('Cari kasus',placeholder='Lowongan, pupuk, beras, WhatsApp…')
        topic=b.selectbox('Topik',['Semua']+sorted({c['topic'] for c in CASES}))
        only=st.checkbox('Tampilkan yang saya simpan')
        results=search_cases(q,CASES,topic)
        if only:results=[c for c in results if c['id'] in st.session_state.saved]
        st.caption(f'{len(results)} kasus · pencarian berdasarkan kata, bukan keputusan benar/salah')
        if not results:st.info('Belum ada kasus yang cocok. Coba kata lain; ketiadaan hasil tidak membuktikan klaim benar.')
        cols=st.columns(3)
        for i,c in enumerate(results):
            with cols[i%3]:card(c,'browse')

elif page=='Lab AI':
    eyebrow('EKSPERIMEN TERBUKA • MODEL BISA KELIRU')
    st.title('Kamu menilai. AI membandingkan.')
    st.write('Tempel artikel utuh berbahasa Indonesia. Lihat prediksi dan pola kata yang memengaruhinya, lalu cari rujukan.')
    st.caption('Teks diproses oleh aplikasi ini; tidak dikirim ke layanan chatbot. Catatan dalam sesi bukan penyimpanan permanen. Jangan masukkan data pribadi.')
    with st.form('ai_input'):
        text=st.text_area('Artikel yang ingin dipelajari',height=200,max_chars=30000,placeholder='Tempel isi artikel, bukan hanya judul atau tautan…')
        a,b=st.columns(2)
        assessment=a.radio('Penilaian awalmu',['Cenderung Hoax','Cenderung Valid','Belum yakin'],index=None)
        confidence=b.slider('Keyakinan pribadi (bukan skor AI)',0,100,50,format='%d%%')
        reason=st.text_input('Alasan singkatmu',max_chars=2000)
        send=st.form_submit_button('Bandingkan penilaian →',type='primary')
    if send:
        st.session_state.pop('ai_result',None)
        if not text.strip() or assessment is None or not reason.strip():st.warning('Lengkapi artikel, penilaian, dan alasanmu.')
        elif len(text.split())<40:
            st.warning('Masukan terlalu singkat untuk alur artikel ini. Tempel artikel lengkap (minimal 40 kata). Batas panjang ini hanya penyaring masukan, bukan jaminan keandalan.')
        else:
            try:
                model=load_model();cleaned=clean(text)
                vec=model.named_steps['tfidf'];x=vec.transform([cleaned]);clf=model.steps[-1][1]
                if x.nnz==0:st.warning('Teks tidak memiliki fitur yang dikenali model; prediksi tidak ditampilkan.')
                else:
                    pred=str(model.predict([cleaned])[0]);names=vec.get_feature_names_out()
                    direction=1 if pred==str(clf.classes_[1]) else -1
                    contributions=[(str(names[j]),float(v*clf.coef_[0,j]*direction)) for j,v in zip(x.indices,x.data)]
                    terms=sorted([x for x in contributions if x[1]>0],key=lambda x:x[1],reverse=True)[:8]
                    result={'text':text,'prediction':pred,'assessment':assessment,'reason':reason,'confidence':confidence,'terms':terms}
                    st.session_state.ai_result=result
                    st.session_state.ai_history.append({k:v for k,v in result.items() if k not in ('text','terms')})
            except Exception as e:
                st.error('Model belum dapat dimuat. Periksa kesesuaian file model dan versi dependensi. Menu belajar tetap dapat dipakai.')
                with st.expander('Detail untuk pengelola'):st.code(str(e))
    result=st.session_state.get('ai_result')
    if result:
        st.caption('Hasil kiriman terakhir. Jika isian di atas diubah, tekan tombol bandingkan lagi.')
        a,b=st.columns(2);a.metric('Penilaianmu',result['assessment']);b.metric('Prediksi SVM',result['prediction'])
        st.write('**Alasanmu:** '+result['reason'])
        if result['assessment']=='Belum yakin':msg='Kamu menunda kesimpulan. Gunakan kesempatan ini untuk mencari bukti.'
        elif result['assessment']=='Cenderung '+result['prediction']:msg='Penilaianmu sejalan dengan model. Kesepakatan bukan bukti kebenaran.'
        else:msg='Penilaianmu berbeda dari model. Model dapat keliru; bandingkan dengan bukti.'
        st.info(msg)
        st.warning('Hasil ini prediksi pola bahasa, bukan pemeriksaan fakta. AI tidak menilai kualitas alasanmu atau menelusuri internet.')
        tab1,tab2,tab3=st.tabs(['Pola kata model','Rujukan terkait','Teks yang dianalisis'])
        with tab1:
            if result['terms']:
                st.bar_chart(pd.DataFrame(result['terms'],columns=['Frasa','Kontribusi']).set_index('Frasa'),color='#8B72DA',horizontal=True)
            st.caption('Kontribusi linear TF-IDF × bobot SVM ke kelas prediksi, sebelum bias. Bukan persentase keyakinan dan bukan alasan faktual. Frasa dapat tumpang tindih.')
        with tab2:
            matches=search_cases(result['text'],CASES)[:3]
            st.caption('Urutan berdasarkan kemiripan kata. Cocokkan objek, waktu, kejadian, dan klaim sebelum memakai rujukan.')
            if not matches:st.info('Belum ada rujukan dengan kata yang cocok dalam koleksi.')
            for c in matches:
                with st.expander(c['title']):st.write(c['claim']);st.write(c['evidence']);reference(c)
        with tab3:st.write(result['text'])
    with st.expander('Seberapa baik model ini dalam pengujian?'):
        a,b=st.columns(2);a.metric('Akurasi uji',f'{METRICS["akurasi"]:.2%}');b.metric('F1-macro',f'{METRICS["f1_macro"]:.4f}')
        st.dataframe(pd.DataFrame(METRICS['confusion_matrix'],index=['Rujukan Hoax','Rujukan Valid'],columns=['Prediksi Hoax','Prediksi Valid']))
        st.write('442 artikel latih, 108 artikel uji. Model salah pada 29 artikel uji. Kelompok kemiripan dipisahkan; 50 kandidat konflik label dikeluarkan sementara dari 600 data awal.')
        st.caption('Satu pembagian uji dataset lama, belum evaluasi semua topik atau peningkatan literasi. Model tidak dilatih ulang dalam paket ini.')
        st.link_button('Dataset dan atribusi ↗','https://data.mendeley.com/datasets/p3hfgr5j3m/1')
        st.caption('Faisal Rahutomo, Inggrid Yanuar, Rosa Andrie Asmara (2018). CC BY 4.0. Data dibersihkan dan difilter untuk eksperimen.')

elif page=='Misi Literasi':
    from missions import MISSIONS
    eyebrow('8 MISI • LATIH CARA BERPIKIR')
    st.title('Bukan cepat menjawab. Tepat menalar.')
    st.caption('Skenario berikut ditulis sebagai ilustrasi pembelajaran, bukan laporan kejadian nyata atau alat ukur psikometrik.')
    done=st.session_state.quiz_answers
    st.progress(len(done)/len(MISSIONS),text=f'{len(done)} dari {len(MISSIONS)} misi selesai dalam sesi ini')
    index=st.selectbox('Pilih misi',range(len(MISSIONS)),format_func=lambda i:f'{i+1:02d} · {MISSIONS[i]["title"]}')
    m=MISSIONS[index]
    a,b=st.columns([1.4,1],gap='large')
    with b:art(m['art'])
    with a:
        st.markdown(f'<span class="tag">{esc(m["skill"])}</span>',unsafe_allow_html=True)
        st.header(m['title']);st.write(m['scenario'])
        with st.form('mission_'+str(index)):
            answer=st.radio('Apa tindakan paling tepat?',m['options'],index=None)
            send=st.form_submit_button('Buka pembahasan →',type='primary')
        if send:
            if answer is None:st.warning('Pilih jawaban terlebih dahulu.')
            else:
                # First attempt only; retries cannot inflate the progress score.
                if str(index) not in done:done[str(index)]={'answer':answer,'correct':answer==m['options'][m['correct']]}
                st.session_state['last_answer_'+str(index)]=answer
                st.rerun()
        if str(index) in done:
            chosen=st.session_state.get('last_answer_'+str(index),done[str(index)]['answer'])
            if chosen==m['options'][m['correct']]:st.success('Pilihanmu tepat untuk situasi ini.')
            else:st.info('Perhatikan kembali langkah pemeriksaannya.')
            st.write('**Mengapa?** '+m['explanation'])
            st.caption('Progres mencatat jawaban pertama. Kamu tetap boleh mencoba lagi untuk belajar.')
            st.link_button('Bacaan pendamping ↗',m['url'])

elif page=='Progres':
    eyebrow('CATATAN BELAJARMU • SESI INI')
    st.title('Apa yang berubah setelah menelisik?')
    a,b,c=st.columns(3)
    a.metric('Misi selesai',f'{len(st.session_state.quiz_answers)}/8')
    b.metric('Refleksi kasus',len(st.session_state.journal))
    c.metric('Kasus disimpan',len(st.session_state.saved))
    correct=sum(x['correct'] for x in st.session_state.quiz_answers.values())
    st.write(f'Jawaban pertama yang tepat: **{correct} dari {len(st.session_state.quiz_answers)} misi yang dikerjakan**.')
    st.caption('Ini catatan aktivitas latihan, bukan ukuran menyeluruh tingkat literasi. Sesi bisa hilang saat halaman dimuat ulang atau koneksi berakhir; unduh catatan sebelum menutup web.')
    for r in st.session_state.journal.values():
        with st.expander(r['title']):
            st.write('**Awal:** '+r['choice']);st.write(r['reason']);st.write('**Sesudah:** '+r['after']);st.write(r['reflection'])
    export={'refleksi':st.session_state.journal,'misi_jawaban_pertama':st.session_state.quiz_answers,'kasus_disimpan':st.session_state.saved,'riwayat_prediksi_tanpa_artikel':st.session_state.ai_history}
    st.download_button('Unduh catatan belajar (.json)',json.dumps(export,ensure_ascii=False,indent=2),file_name='catatan_telisik.json',mime='application/json',type='primary')
    st.divider()
    with st.expander('Tentang TELISIK & sumber konten'):
        st.write('TELISIK adalah prototipe media pembelajaran literasi kritis. Tidak berafiliasi dengan ANTARA, MAFINDO, atau penerbit sumber. Ringkasan kasus ditulis ulang dan ditautkan ke pemeriksaan asli. Ilustrasi vektor adalah dekorasi, bukan bukti.')
        st.write('Enam arsip kasus dipilih secara manual, termasuk empat pemeriksaan bertanggal 6 Oktober 2026. Semua contoh kasus ini membahas klaim bermasalah; jangan gunakan sebagai dataset dua kelas.')
        st.write('Radar mengambil RSS publik ANTARA saat halaman dipakai, dengan cache 15 menit. Tidak ada proses pembaruan di latar belakang jika web tidak digunakan. Tidak ada model generatif atau pelatihan otomatis dari RSS.')
        st.link_button('Tentang RSS ANTARA','https://www.antaranews.com/rss')
        st.write('Kebaruan yang diusulkan: siklus penilaian awal → bukti → refleksi, didampingi model transparan. Efektivitas pembelajaran masih perlu diuji kepada pengguna.')
    reset=st.checkbox('Saya ingin menghapus catatan sesi ini')
    if st.button('Hapus progres sesi',disabled=not reset):
        for k in list(st.session_state):
            if k!='nav':del st.session_state[k]
        st.rerun()

st.markdown('<div class="footer"><strong>✳ TELISIK / Berani cek. Bijak percaya.</strong><span>Prototipe pendidikan · Prediksi bukan vonis · Sumber tetap utama</span></div>',unsafe_allow_html=True)
