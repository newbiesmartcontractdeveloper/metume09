#!/usr/bin/env python3
"""GPX sürüş analizi: hız/irtifa istatistikleri, güç tahmini, kurulum senaryoları.

Güçölçer olmadan, GPX'teki konum+zaman+irtifadan fizik modeliyle güç tahmin eder:

    P = (F_aero + F_yuvarlanma + F_yerçekimi) * v / eta
    F_aero  = 0.5 * rho * CdA * v^2
    F_yuv.  = Crr * m * g * cos(theta)
    F_yerç. = m * g * sin(theta)

Kullanım:
    python3 tools/ride_analysis.py Morning_Ride.gpx --kilo 75 --bisiklet 15
"""
import argparse
import datetime as dt
import xml.etree.ElementTree as ET

import numpy as np

NS = "{http://www.topografix.com/GPX/1/1}"
G = 9.80665
ETA = 0.95  # aktarma verimi


def read_gpx(path):
    pts = []
    for p in ET.parse(path).getroot().iter(NS + "trkpt"):
        ele = p.find(NS + "ele")
        tm = p.find(NS + "time")
        if tm is None:
            continue
        pts.append((
            dt.datetime.fromisoformat(tm.text.replace("Z", "+00:00")),
            float(p.get("lat")),
            float(p.get("lon")),
            float(ele.text) if ele is not None else 0.0,
        ))
    if len(pts) < 2:
        raise SystemExit(f"{path}: yeterli trkpt yok")
    return pts


def haversine_cumulative(lat, lon):
    r = 6371000.0
    la, lo = np.radians(lat), np.radians(lon)
    a = (np.sin(np.diff(la) / 2) ** 2
         + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2)
    return np.concatenate([[0.0], np.cumsum(2 * r * np.arcsin(np.sqrt(a)))])


def smooth(x, w):
    pad = np.pad(x, (w // 2, w // 2), mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")[:len(x)]


def air_density(altitude_m, temp_c):
    """Barometrik yükseklik formülü + ideal gaz."""
    p = 101325 * (1 - 2.25577e-5 * altitude_m) ** 5.25588
    return p / (287.05 * (temp_c + 273.15)), p


def find_gps_dropouts(ts, cum, min_freeze=20.0, jump_speed=55.0):
    """Sinyal kaybı: konum donuyor, sonra tek örnekte sıçrıyor.

    Bu bölümler hem sahte maksimum hız hem de sahte 'durak' üretir.
    """
    dtv = np.diff(ts)
    seg = np.diff(cum)
    speed = seg / np.maximum(dtv, 1e-9)
    bad = []
    for i in range(len(seg)):
        if speed[i] * 3.6 > jump_speed:
            j = i
            while j > 0 and seg[j - 1] == 0.0:
                j -= 1
            if ts[i] - ts[j] >= min_freeze or dtv[i] >= 8:
                bad.append((ts[j], ts[i + 1]))
    return bad


class Ride:
    def __init__(self, path, mass_rider, mass_bike, temp_c, cda, crr):
        pts = read_gpx(path)
        t0 = pts[0][0]
        ts = np.array([(p[0] - t0).total_seconds() for p in pts])
        lat = np.array([p[1] for p in pts])
        lon = np.array([p[2] for p in pts])
        ele = np.array([p[3] for p in pts])
        cum = haversine_cumulative(lat, lon)

        self.start = t0
        self.dropouts = find_gps_dropouts(ts, cum)

        # 1 Hz'e yeniden örnekle: Strava "smart recording" düzensiz aralık yazar
        self.t = np.arange(0, ts[-1] + 1, 1.0)
        self.d = np.interp(self.t, ts, cum)
        self.ele = smooth(np.interp(self.t, ts, ele), 31)
        self.v = smooth(np.gradient(self.d, self.t), 9)
        self.grade = np.clip(
            smooth(np.where(self.v > 0.5,
                            np.gradient(self.ele, self.t) / np.maximum(self.v, 0.5),
                            0.0), 21),
            -0.15, 0.15)
        self.theta = np.arctan(self.grade)

        self.bad = np.zeros_like(self.t, dtype=bool)
        for a, b in self.dropouts:
            self.bad |= (self.t >= a) & (self.t <= b)

        self.mass = mass_rider + mass_bike
        self.mass_rider = mass_rider
        self.cda, self.crr = cda, crr
        self.rho, self.pressure = air_density(self.ele.mean(), temp_c)

        self.power = self.steady_power(self.v, cda, crr, self.mass)
        self.moving = (self.v > 1.0) & ~self.bad
        self.powered = self.moving & (self.power > 5.0)

    def steady_power(self, v, cda, crr, mass):
        f = (0.5 * self.rho * cda * v ** 2
             + crr * mass * G * np.cos(self.theta)
             + mass * G * np.sin(self.theta))
        return f * v / ETA

    def solve_speed(self, power, cda, crr, mass):
        """0.5*rho*CdA*v^3 + (Crr*m*g*cos + m*g*sin)*v = P*eta  (Newton)"""
        a = 0.5 * self.rho * cda
        b = crr * mass * G * np.cos(self.theta) + mass * G * np.sin(self.theta)
        c = power * ETA
        v = np.maximum(self.v, 0.5)
        for _ in range(80):
            v = np.clip(v - (a * v ** 3 + b * v - c) / np.maximum(3 * a * v ** 2 + b, 1e-6),
                        0.3, 25.0)
        return v

    def scenario_time(self, cda, crr, mass):
        """Aynı güç profili, farklı kurulum -> yeni toplam süre.

        Pedal basılmayan örnekler (iniş, fren, duraklama) değiştirilmez;
        bu tahmini muhafazakâr tarafta tutar.
        """
        vnew = np.where(self.powered,
                        np.maximum(self.solve_speed(self.power, cda, crr, mass), 0.5),
                        self.v)
        step = np.gradient(self.d, self.t)
        return np.where(self.powered, step / np.maximum(vnew, 0.3), 1.0).sum()

    def energy_split(self):
        m, v = self.powered, self.v
        aero = (0.5 * self.rho * self.cda * v ** 2 * v)[m].sum()
        roll = (self.crr * self.mass * G * np.cos(self.theta) * v)[m].sum()
        climb = (np.maximum(self.mass * G * np.sin(self.theta), 0) * v)[m].sum()
        return aero, roll, climb


def hms(seconds):
    return str(dt.timedelta(seconds=int(seconds)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpx")
    ap.add_argument("--kilo", type=float, default=75.0, help="sürücü ağırlığı (kg)")
    ap.add_argument("--bisiklet", type=float, default=15.0, help="bisiklet+ekipman (kg)")
    ap.add_argument("--sicaklik", type=float, default=18.0, help="hava sıcaklığı (C)")
    ap.add_argument("--cda", type=float, default=0.45)
    ap.add_argument("--crr", type=float, default=0.0080)
    args = ap.parse_args()

    r = Ride(args.gpx, args.kilo, args.bisiklet, args.sicaklik, args.cda, args.crr)

    print("=== GENEL ===")
    print(f"Başlangıç      : {r.start:%Y-%m-%d %H:%M} UTC")
    print(f"Mesafe / süre  : {r.d[-1]/1000:.2f} km / {hms(r.t[-1])}")
    print(f"Hareket ort.   : {r.v[r.moving].mean()*3.6:.1f} km/h")
    dz = np.diff(r.ele)
    print(f"Tırmanış/iniş  : +{dz[dz>0].sum():.0f} m / -{-dz[dz<0].sum():.0f} m "
          f"(net {r.ele[-1]-r.ele[0]:+.0f} m, {r.ele.min():.0f}-{r.ele.max():.0f} m)")

    if r.dropouts:
        print(f"\n!! GPS sinyal kaybı ({len(r.dropouts)} adet) — analizden çıkarıldı:")
        for a, b in r.dropouts:
            print(f"   t={hms(a)}..{hms(b)}  ({b-a:.0f} s donma + konum sıçraması)")

    print("\n=== HAVA ===")
    print(f"Ort. rakım {r.ele.mean():.0f} m -> {r.pressure/100:.0f} hPa, "
          f"rho = {r.rho:.3f} kg/m3 (deniz seviyesine göre %{(1-r.rho/1.225)*100:.1f} daha seyrek)")

    print(f"\n=== GÜÇ TAHMİNİ (CdA={r.cda:.2f}, Crr={r.crr:.4f}, m={r.mass:.0f} kg) ===")
    p = r.power[r.powered]
    print(f"Pedal basılan  : {hms(r.powered.sum())} (%{r.powered.sum()/len(r.t)*100:.0f})")
    print(f"Ortalama güç   : {p.mean():.0f} W = {p.mean()/r.mass_rider:.2f} W/kg")
    print(f"Toplam iş      : {p.sum()/1000:.0f} kJ (~{p.sum()/1000/4.184/0.24:.0f} kcal)")

    aero, roll, climb = r.energy_split()
    tot = aero + roll + climb
    print("\n=== ENERJİ NEREYE GİTTİ ===")
    for name, val in (("aero", aero), ("yuvarlanma", roll), ("tırmanış", climb)):
        print(f"  {name:<11} {val/1000:6.0f} kJ  %{val/tot*100:.0f}")

    ref = r.scenario_time(r.cda, r.crr, r.mass)
    print("\n=== AYNI EFOR, FARKLI KURULUM ===")
    print(f"  {'senaryo':<44} {'süre':>8}   kazanç")
    print(f"  {'mevcut kurulum':<44} {hms(ref):>8}   referans ({r.d[-1]/ref*3.6:.1f} km/h)")
    for cda, crr, dm, label in [
        (r.cda, 0.0065, 0.0, "doğru basınç (Crr -> .0065)"),
        (0.38, r.crr, 0.0, "pozisyon + dar kıyafet (CdA -> .38)"),
        (r.cda, 0.0045, 0.0, "slick lastik + doğru basınç (Crr -> .0045)"),
        (r.cda, r.crr, -2.0, "rijit çatal / -2 kg"),
        (0.40, 0.0055, 0.0, "ucuz paket (basınç + pozisyon + kıyafet)"),
        (0.38, 0.0045, -2.0, "tam paket (lastik + rijit çatal dahil)"),
    ]:
        t = r.scenario_time(cda, crr, r.mass + dm)
        print(f"  {label:<44} {hms(t):>8}   {(t-ref)/60:+.1f} dk / "
              f"{r.d[-1]/t*3.6 - r.d[-1]/ref*3.6:+.2f} km/h")


if __name__ == "__main__":
    main()
