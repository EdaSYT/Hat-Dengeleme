import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import time
# =========================================================
# 2. NGROK TOKENİNİZİ TANIMLAYIN (BURASI ÇOK ÖNEMLİ)
# =========================================================
from pyngrok import ngrok
# Örn: ngrok.set_auth_token("2abc123XYZ...")
ngrok.set_auth_token("3DLf70ziIi1AZxt0XgqDsIOZWf0_dMcYtE2DAcpXEQPe4ZZU")
# =========================================================
# 3. STREAMLIT UYGULAMA DOSYASININ YAZILMASI
# =========================================================
with open("app.py", "w", encoding="utf-8") as f:
    f.write("""
import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model

st.set_page_config(layout="wide", page_title="Hattı Dengeleme Optimizasyonu")
st.title("🏭 Montaj Hattı Dengeleme & Operatör Atama Sistemi")
st.markdown("Google OR-Tools CP-SAT Solver tabanlı optimizasyon arayüzü.")

I = range(1, 64)
J = range(1, 37)
W = range(1, 37)

t_raw = {
    1: 2.43,  2: 9.79,  3: 2.12,  4: 9.92,  5: 4.66,  6: 11.58,
    7: 1.01,  8: 1.44,  9: 9.66, 10: 10.30, 11: 0.49, 12: 7.13,
    13: 7.18, 14: 2.44, 15: 3.58, 16: 4.90, 17: 3.21, 18: 7.78,
    19: 11.27, 20: 11.35, 21: 0.80, 22: 3.31, 23: 9.83, 24: 0.80,
    25: 4.61, 26: 5.20, 27: 11.89, 28: 6.30, 29: 13.32, 30: 0.98,
    31: 14.20, 32: 6.13, 33: 0.98, 34: 14.49, 35: 3.14, 36: 12.12,
    37: 1.07, 38: 5.14, 39: 5.63, 40: 0.57, 41: 10.13, 42: 0.90,
    43: 1.39, 44: 1.43, 45: 0.51, 46: 10.74, 47: 5.65, 48: 7.38,
    49: 1.71, 50: 15.09, 51: 7.31, 52: 6.93, 53: 10.72, 54: 1.31,
    55: 6.45, 56: 2.39, 57: 0.89, 58: 11.06, 59: 8.02, 60: 6.48,
    61: 3.13, 62: 0.53, 63: 7.74
}

SCALE = 100
t = {i: int(round(t_raw[i] * SCALE)) for i in t_raw}
P = [(i, i + 1) for i in range(1, 63)]
d = {j: {k: 2 * abs(j - k) for k in J} for j in J}
BIG_M = sum(t.values())

st.sidebar.header("🎛️ Model Parametreleri")
L = st.sidebar.number_input("Maksimum Yürüme Mesafesi (L)", value=4)
D = st.sidebar.number_input("Hedef Üretim Miktarı (D)", value=32)
T = st.sidebar.number_input("Vardiya Süresi (T - dk)", value=510)
U_MAX = st.sidebar.slider("Maksimum Operatör Doluluğu (U_MAX)", 0.1, 1.0, 1.0)
time_limit = st.sidebar.slider("Çözücü Zaman Limiti (Saniye)", 5, 60, 15)
st.sidebar.markdown("---")
target_workers = st.sidebar.slider("Detaylandırılacak Operatör Sayısı", 1, 36, 29)

def solve_model(exact_workers=None):
    model = cp_model.CpModel()
    x = {(i, j): model.NewBoolVar(f"x_{i}_{j}") for i in I for j in J}
    y = {(w, j): model.NewBoolVar(f"y_{w}_{j}") for w in W for j in J}
    z = {w: model.NewBoolVar(f"z_{w}") for w in W}
    l = {j: model.NewIntVar(0, BIG_M, f"l_{j}") for j in J}
    q = {(w, j): model.NewIntVar(0, BIG_M, f"q_{w}_{j}") for w in W for j in J}
    C = model.NewIntVar(0, BIG_M, "C")

    for i in I: model.Add(sum(x[i, j] for j in J) == 1)
    for i, h in P: model.Add(sum(j * x[i, j] for j in J) <= sum(j * x[h, j] for j in J))
    for j in J: model.Add(l[j] == sum(t[i] * x[i, j] for i in I))
    for j in J: model.Add(sum(y[w, j] for w in W) == 1)
    for w in W:
        for j in J: model.Add(y[w, j] <= z[w])
    for w in W:
        for j in J:
            model.Add(q[w, j] <= l[j])
            model.Add(q[w, j] <= BIG_M * y[w, j])
            model.Add(q[w, j] >= l[j] - BIG_M * (1 - y[w, j]))
    for w in W: model.Add(sum(q[w, j] for j in J) <= C)
    for j in J: model.Add(l[j] <= C)
    for w in W:
        for j in J:
            for k in J:
                if j < k and d[j][k] > L: model.Add(y[w, j] + y[w, k] <= 1)
    if exact_workers is not None: model.Add(sum(z[w] for w in W) == exact_workers)

    model.Minimize(C)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]: return None
    C_value = solver.Value(C) / SCALE

    solution = {
        "C": C_value,
        "used_workers": sum(solver.Value(z[w]) for w in W),
        "stations_of_worker": {w: [] for w in W},
        "ops_of_station": {j: [] for j in J},
        "station_loads": {j: solver.Value(l[j]) / SCALE for j in J},
        "worker_load_per_product": {w: sum(solver.Value(q[w, j]) for j in J) / SCALE for w in W},
        "worker_load_per_shift": {w: D * sum(solver.Value(q[w, j]) for j in J) / SCALE for w in W},
        "worker_U": {w: 100 * ((D / T) * (sum(solver.Value(q[w, j]) for j in J) / SCALE)) for w in W},
        "reachable_output": T / C_value if C_value > 0 else float("inf"),
        "meets_target": (T / C_value >= D - 1e-6) if C_value > 0 else True,
    }
    for i in I:
        for j in J:
            if solver.Value(x[i, j]) == 1: solution["ops_of_station"][j].append(i)
    for w in W:
        for j in J:
            if solver.Value(y[w, j]) == 1: solution["stations_of_worker"][w].append(j)
    return solution

if st.button("🚀 Modeli Çöz ve Senaryoları Üret"):
    with st.spinner("Hesaplanıyor..."):
        results = {}
        scenarios_to_run = sorted(list(set([1, target_workers-1, target_workers, target_workers+1, 36])))
        scenarios_to_run = [s for s in scenarios_to_run if 1 <= s <= 36]
        for eps in scenarios_to_run: results[eps] = solve_model(exact_workers=eps)

        summary_data = []
        for eps, res in results.items():
            if res is None: summary_data.append([eps, "Infeasible", "-", "-", "Hayır"])
            else: summary_data.append([eps, f"{res['C']:.2f}", res['used_workers'], f"{res['reachable_output']:.2f}", "Evet" if res['meets_target'] else "Hayır"])
        df_summary = pd.DataFrame(summary_data, columns=["Operatör (Epsilon)", "Çevrim Süresi (C)", "Kullanılan Op.", "Ulaşılabilir Üretim", "Hedef Sağlandı mı?"])
        st.subheader("📊 Senaryo Özet Tablosu")
        st.dataframe(df_summary, use_container_width=True)

        if target_workers in results and results[target_workers] is not None:
            res = results[target_workers]
            st.success(f"🎯 Detaylı Senaryo Raporu: {target_workers} Operatör")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Çevrim Süresi (C)", f"{res['C']:.2f} dk")
            kpi2.metric("Kullanılan Operatör", f"{res['used_workers']} Kişi")
            kpi3.metric("Ulaşılabilir Üretim", f"{res['reachable_output']:.2f} Adet")
            kpi4.metric("Hedef Karşılanıyor mu?", "Evet" if res['meets_target'] else "Hayır")

            tab1, tab2 = st.tabs(["📋 İstasyon Yükleri", "🧑‍🔧 Operatör Detayları"])
            with tab1:
                station_data = [[j, str(res['ops_of_station'][j]), f"{res['station_loads'][j]:.2f} dk"] for j in J]
                st.dataframe(pd.DataFrame(station_data, columns=["İstasyon No", "Atanan Operasyonlar", "İstasyon Yükü"]), use_container_width=True, hide_index=True)
            with tab2:
                worker_data = [[w, str(res["stations_of_worker"][w]), f"{res['worker_load_per_product'][w]:.2f} dk", f"%{res['worker_U'][w]:.2f}"] for w in W if len(res["stations_of_worker"][w]) > 0]
                st.dataframe(pd.DataFrame(worker_data, columns=["Operatör No", "İstasyonlar", "Ürün Başı Yük", "U Oranı"]), use_container_width=True, hide_index=True)
""")
# =========================================================
# 4. STREAMLIT'I BAŞLAT VE KESİNTİSİZ NGROK LİNKİNİ AL
# =========================================================
import os
os.system("streamlit run app.py --server.port 8501 &")

# Güvenli tünel linkini oluşturuyoruz
public_url = ngrok.connect(8501, proto="http")
print("\n🎉 ÇİLE BİTTİ! UYGULAMANIZ HATA VERMEDEN HAZIR:")
print(public_url.public_url)
print("---------------------------------------------------\n")
