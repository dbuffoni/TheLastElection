"""Self-contained HTML report of the population's evolution in one run (spec 10.3: run explorer, citizen panel,
network, stability). No server and no plotting dependency: data are embedded as JSON and drawn with a small inline
script (line charts with a hover crosshair, ideology scatter per snapshot day, histograms, tables)."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import polars as pl
from tle.analysis import population as pop
from tle.registry.metrics import load_metrics
from tle.runs import Run

GROUP_TITLES = {
    "population": "Demography",
    "ideology": "Ideology and polarization",
    "opinion": "Opinion, fear and silence",
    "wealth": "Wealth and classes",
    "network": "Social network",
}


def _clean(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def _series(df: pl.DataFrame, col: str) -> list:
    return [_clean(float(x)) if x is not None else None for x in df[col].to_list()]


def _scatter_frames(run: Run, max_days: int = 6, max_points: int = 600) -> list[dict]:
    days = run.panel_days()
    if not days:
        return []
    pick = sorted({days[int(round(i))] for i in np.linspace(0, len(days) - 1, min(max_days, len(days)))})
    out = []
    for d in pick:
        p = run.panel(d).filter(pl.col("age") >= 18)
        if len(p) > max_points:
            p = p.sample(max_points, seed=0)
        out.append(
            {
                "day": float(d),
                "x": [round(v, 3) for v in p["I1"].to_list()],
                "y": [round(v, 3) for v in p["I2"].to_list()],
                "c": p["cls"].to_list(),
                "n": len(p),
            }
        )
    return out


def _hist_frames(run: Run, col: str = "I1", bins: int = 24, max_days: int = 6) -> list[dict]:
    days = run.panel_days()
    if not days:
        return []
    pick = sorted({days[int(round(i))] for i in np.linspace(0, len(days) - 1, min(max_days, len(days)))})
    out = []
    for d in pick:
        x = run.panel(d).filter(pl.col("age") >= 18)[col].to_numpy()
        h, _ = np.histogram(x, bins=bins, range=(-1, 1))
        out.append({"day": float(d), "h": (h / max(1, h.sum())).round(4).tolist()})
    return out


def build_report(run: Run) -> str:
    specs = load_metrics()
    man, st = run.manifest, run.status
    game = run.metrics().df
    warm = run.warmup().df
    cols = [c for c in specs if c in game.columns]
    charts = []
    for mid in cols:
        s = specs[mid]
        t = s.target
        band = None
        if t and t.kind == "band":
            band = [t.lo, t.hi]
        elif t and t.kind in ("min",):
            band = [t.value, None]
        elif t and t.kind in ("max",):
            band = [None, t.value]
        if t and t.scenarios and man.get("scenario") not in t.scenarios:
            band = None
        charts.append(
            {
                "id": mid,
                "title": s.title,
                "unit": s.unit,
                "group": s.group,
                "spec": ", ".join(s.spec),
                "band": band,
                "band_from": (t.after_days if t else 0),
                "x": game["day"].to_list(),
                "y": _series(game, mid),
                "wx": warm["day"].to_list() if len(warm) and mid in warm.columns else [],
                "wy": _series(warm, mid) if len(warm) and mid in warm.columns else [],
            }
        )
    ms = run.milestones()
    marks = (
        [
            {"day": r["day"], "kind": r["kind"], "what": r["what"], "value": r["value"]}
            for r in ms.iter_rows(named=True)
        ]
        if len(ms)
        else []
    )
    targets = pop.evaluate_targets(run, specs)
    stab = pop.stability_table(run)
    mob = pop.mobility(run, 0.0, None)
    flows = pop.class_flows(run, 0.0, None)
    ident = pop.scenario_identity(run)
    rates = pop.vital_rates(run)
    days = run.panel_days()
    age0, age1 = (
        (pop.age_distribution(run, days[0]), pop.age_distribution(run, days[-1]))
        if days
        else (pl.DataFrame(), pl.DataFrame())
    )
    deg0, deg1 = (
        (pop.degree_distribution(run, days[0]), pop.degree_distribution(run, days[-1]))
        if days
        else (pl.DataFrame(), pl.DataFrame())
    )
    data = {
        "charts": charts,
        "marks": marks,
        "scatter": _scatter_frames(run),
        "hist": _hist_frames(run),
        "age": [
            {"day": days[0], "rows": age0.rows()} if len(age0) else None,
            {"day": days[-1], "rows": age1.rows()} if len(age1) else None,
        ]
        if days
        else [],
        "deg": [
            {"day": days[0], "rows": deg0.rows()} if len(deg0) else None,
            {"day": days[-1], "rows": deg1.rows()} if len(deg1) else None,
        ]
        if days
        else [],
        "groups": GROUP_TITLES,
    }
    checks = run.checks()
    head = (
        f"{html.escape(str(man.get('scenario', '')))} · {html.escape(str(man.get('label', '')))} · seed {man.get('seed')} · "
        f"{st.get('state')} · day {st.get('day', 0):.0f} · outcome {html.escape(str(st.get('outcome') or 'running'))}"
    )

    def table(df: pl.DataFrame, title: str) -> str:
        if not len(df):
            return ""
        th = "".join(f"<th>{html.escape(c)}</th>" for c in df.columns)
        rows = ""
        for r in df.iter_rows():
            rows += "<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in r) + "</tr>"
        return f"<h3>{html.escape(title)}</h3><div class='tw'><table><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>"

    def chips(df: pl.DataFrame) -> str:
        if not len(df):
            return "<p class='muted'>No target metrics in this run.</p>"
        icon = {"ok": "✓", "miss": "✕", "n/a": "–"}
        rows = ""
        for r in df.iter_rows(named=True):
            rows += (
                f"<tr><td>{html.escape(r['metric'])}</td><td>{html.escape(r['expected'])}</td><td>{_fmt(r['observed'])}</td>"
                f"<td><span class='chip {r['verdict'].replace('/', '')}'>{icon[r['verdict']]} {r['verdict']}</span></td>"
                f"<td class='muted'>{html.escape(r['note'])}</td></tr>"
            )
        return (
            "<div class='tw'><table><thead><tr><th>metric</th><th>expected</th><th>observed</th><th>verdict</th><th>note</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    facts = []
    if mob:
        facts.append(
            (
                "Ideology label changed",
                f"{mob['share_label_changed']:.0%}",
                f"of {mob['people']} adults, day {mob['day0']:.0f}→{mob['day1']:.0f} (target 15–35% at 2 years)",
            )
        )
        facts.append(
            ("Median displacement", f"{mob['median_displacement']:.3f}", "bipolar units (target 0.17–0.50)")
        )
    if rates:
        facts.append(
            (
                "Birth / death rate",
                f"{rates['birth_rate']:.1%} / {rates['death_rate']:.1%}",
                f"per year; net growth {rates['net_growth']:.1%}",
            )
        )
    if ident:
        facts.append(
            (
                "Polarization kept through warm-up",
                f"{ident['polarization_retained']:.0%}",
                f"Sarle {ident['sarle_start']:.2f} → {ident['sarle_day0']:.2f} (camps if > 0.555)",
            )
        )
    fact_html = "".join(
        f"<div class='fact'><div class='fl'>{html.escape(a)}</div><div class='fv'>{html.escape(b)}</div><div class='fn'>{html.escape(c)}</div></div>"
        for a, b, c in facts
    )
    chk = (
        "<span class='chip ok'>✓ all invariants hold</span>"
        if checks.get("ok")
        else "<span class='chip miss'>✕ "
        + ", ".join(f"{k} (day {v:.0f})" for k, v in checks["first_violation_day"].items())
        + "</span>"
    )
    body = (
        _TEMPLATE.replace("__TITLE__", html.escape(man.get("run_id", run.id)))
        .replace("__HEAD__", head)
        .replace("__FACTS__", fact_html)
        .replace("__CHECKS__", chk)
        .replace("__TARGETS__", chips(targets))
        .replace(
            "__TABLES__",
            table(stab, "Stability verdict per metric (last 180 days)")
            + table(flows, "Class flows, day 0 → last snapshot"),
        )
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
    )
    return body


def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return "–" if np.isnan(v) else f"{v:.3f}"
    return html.escape(str(v))


def write_report(run: Run, out: str | Path) -> Path:
    out = Path(out)
    out.write_text(build_report(run), encoding="utf-8")
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Population report __TITLE__</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#0ca30c;--crit:#d03b3b;--okt:#006300;--misst:#b02a2a;--band:rgba(42,120,214,.13)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--okt:#0ca30c;--misst:#ec6b6b;--band:rgba(57,135,229,.18)}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--okt:#0ca30c;--misst:#ec6b6b;--band:rgba(57,135,229,.18)}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 10px}h3{font-size:13px;margin:18px 0 6px;color:var(--ink2)}
.muted{color:var(--muted)}.sub{color:var(--ink2);margin-bottom:16px}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:12px 0}
.fact{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:12px 14px}.fl{color:var(--ink2);font-size:12px}.fv{font-size:26px;font-weight:600;margin:2px 0}.fn{color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:10px 12px;position:relative}
.card .t{font-size:13px;font-weight:600}.card .v{float:right;font-variant-numeric:tabular-nums;color:var(--ink2)}.card .u{color:var(--muted);font-size:11px}
svg{display:block;width:100%;height:auto;overflow:visible}svg text{fill:var(--muted);font-size:10px}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:6px;padding:4px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.18);display:none;z-index:9;white-space:nowrap}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}th{color:var(--ink2);font-weight:600}
.chip{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;border:1px solid currentColor}.chip.ok{color:var(--okt)}.chip.miss{color:var(--misst)}.chip.na{color:var(--muted)}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--ink2);font-size:12px;margin:6px 0}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
button{background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:6px;padding:3px 10px;cursor:pointer;float:right}
</style></head><body><main>
<button onclick="var r=document.documentElement;r.dataset.theme=(r.dataset.theme==='dark'||(!r.dataset.theme&&matchMedia('(prefers-color-scheme:dark)').matches))?'light':'dark'">Theme</button>
<h1>Population report · __TITLE__</h1><div class="sub">__HEAD__ · __CHECKS__</div>
<div class="facts">__FACTS__</div>
<h2>Target scorecard</h2>__TARGETS__
<div id="sections"></div>
<h2>Citizens over time</h2>
<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>lower class</span><span><i class="sw" style="background:var(--s2)"></i>middle class</span><span><i class="sw" style="background:var(--s3)"></i>upper class</span><span class="muted">each panel: ideology axis 1 (x) vs axis 2 (y), adults, one snapshot day</span></div>
<div class="grid" id="scatter"></div>
<h3>Distribution of ideology axis 1 over snapshot days (share of adults per bin)</h3><div class="card"><div id="ridge"></div></div>
<h2>Age structure and network</h2>
<div class="grid"><div class="card"><div class="t">Age distribution</div><div class="legend"><span><i class="sw" style="background:var(--muted)"></i>first snapshot</span><span><i class="sw" style="background:var(--s1)"></i>last snapshot</span></div><div id="age"></div></div>
<div class="card"><div class="t">Degree distribution</div><div class="legend"><span><i class="sw" style="background:var(--muted)"></i>first snapshot</span><span><i class="sw" style="background:var(--s1)"></i>last snapshot</span></div><div id="deg"></div></div></div>
__TABLES__
<div class="tip" id="tip"></div>
<script>
const D=__DATA__;const NS='http://www.w3.org/2000/svg';const tip=document.getElementById('tip');
function el(n,a,p){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function fmt(v){if(v==null)return '–';const a=Math.abs(v);return a>=1000?v.toFixed(0):a>=10?v.toFixed(1):a>=1?v.toFixed(2):v.toFixed(3)}
function showTip(t,ev){tip.textContent=t;tip.style.display='block';tip.style.left=Math.min(innerWidth-tip.offsetWidth-8,ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px'}
function hideTip(){tip.style.display='none'}
function line(host,c){const W=320,H=116,m={l:40,r:6,t:6,b:20};const xs=c.wx.concat(c.x),ys=c.wy.concat(c.y);
 const ok=ys.filter(v=>v!=null);if(!ok.length){host.innerHTML='<div class="muted">no data</div>';return}
 let lo=Math.min(...ok),hi=Math.max(...ok);if(c.band){for(const b of c.band){if(b!=null){lo=Math.min(lo,b);hi=Math.max(hi,b)}}}
 if(hi-lo<1e-9){hi=lo+1e-6+Math.abs(lo)*0.05}const pad=(hi-lo)*0.08;const nonneg=Math.min(...ok)>=0&&!(c.band&&c.band.some(b=>b!=null&&b<0));lo=nonneg?Math.max(0,lo-pad):lo-pad;hi+=pad;
 const x0=Math.min(...xs),x1=Math.max(...xs);const X=v=>m.l+(v-x0)/(x1-x0||1)*(W-m.l-m.r),Y=v=>m.t+(1-(v-lo)/(hi-lo))*(H-m.t-m.b);
 const s=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':c.title});
 for(const f of [0,.5,1]){const v=lo+(hi-lo)*f;el('line',{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),stroke:'var(--grid)','stroke-width':1},s);const t=el('text',{x:m.l-4,y:Y(v)+3,'text-anchor':'end'},s);t.textContent=fmt(v)}
 if(x0<0){el('line',{x1:X(0),x2:X(0),y1:m.t,y2:H-m.b,stroke:'var(--axis)','stroke-dasharray':'3 3'},s);const t=el('text',{x:X(0)-3,y:m.t+8,'text-anchor':'end'},s);t.textContent='warm-up'}
 const t0=el('text',{x:Math.max(X(x0),m.l+2),y:H-4},s);t0.textContent=Math.round(x0)+' d';const t1=el('text',{x:X(x1),y:H-4,'text-anchor':'end'},s);t1.textContent=Math.round(x1)+' d';
 if(c.band){const a=c.band[0]==null?lo:c.band[0],b=c.band[1]==null?hi:c.band[1];const bx=Math.max(c.band_from||0,0);
  el('rect',{x:X(Math.max(bx,x0)),y:Y(Math.min(b,hi)),width:Math.max(0,X(x1)-X(Math.max(bx,x0))),height:Math.max(0,Y(Math.max(a,lo))-Y(Math.min(b,hi))),fill:'var(--band)'},s)}
 const path=(px,py,dash)=>{let d='',pen=false;for(let i=0;i<px.length;i++){if(py[i]==null){pen=false;continue}d+=(pen?'L':'M')+X(px[i]).toFixed(1)+' '+Y(py[i]).toFixed(1);pen=true}
  if(d)el('path',{d,fill:'none',stroke:'var(--s1)','stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round',opacity:dash?.45:1,'stroke-dasharray':dash?'4 3':'none'},s)};
 path(c.wx,c.wy,true);path(c.x,c.y,false);
 for(const mk of D.marks){if(mk.day<x0||mk.day>x1)continue;el('line',{x1:X(mk.day),x2:X(mk.day),y1:m.t,y2:H-m.b,stroke:mk.kind==='pillar_level'?'var(--s2)':'var(--muted)','stroke-width':1,'stroke-dasharray':'1 3',opacity:.8},s)}
 const cross=el('line',{y1:m.t,y2:H-m.b,stroke:'var(--axis)','stroke-width':1,visibility:'hidden'},s);const dot=el('circle',{r:4,fill:'var(--s1)',stroke:'var(--surface)','stroke-width':2,visibility:'hidden'},s);
 const hit=el('rect',{x:m.l,y:0,width:W-m.l-m.r,height:H,fill:'transparent'},s);
 hit.addEventListener('mousemove',ev=>{const r=s.getBoundingClientRect();const px=(ev.clientX-r.left)/r.width*W;const day=x0+(px-m.l)/(W-m.l-m.r)*(x1-x0);
  let bi=0,bd=1e18;for(let i=0;i<xs.length;i++){const d=Math.abs(xs[i]-day);if(ys[i]!=null&&d<bd){bd=d;bi=i}}
  cross.setAttribute('x1',X(xs[bi]));cross.setAttribute('x2',X(xs[bi]));cross.setAttribute('visibility','visible');dot.setAttribute('cx',X(xs[bi]));dot.setAttribute('cy',Y(ys[bi]));dot.setAttribute('visibility','visible');
  const near=D.marks.filter(mk=>Math.abs(mk.day-xs[bi])<8).map(mk=>' · '+mk.what+(mk.kind==='pillar_level'?' L'+mk.value:'')).join('');
  showTip(c.title+': '+fmt(ys[bi])+' ('+(xs[bi]<0?'warm-up ':'')+'day '+Math.round(xs[bi])+')'+near,ev)});
 hit.addEventListener('mouseleave',()=>{cross.setAttribute('visibility','hidden');dot.setAttribute('visibility','hidden');hideTip()});
 host.appendChild(s)}
const sections=document.getElementById('sections');
for(const g of Object.keys(D.groups)){const cs=D.charts.filter(c=>c.group===g);if(!cs.length)continue;
 const h=document.createElement('h2');h.textContent=D.groups[g];sections.appendChild(h);const grid=document.createElement('div');grid.className='grid';sections.appendChild(grid);
 for(const c of cs){const card=document.createElement('div');card.className='card';const last=[...c.y].reverse().find(v=>v!=null);
  card.innerHTML=`<div class="t">${c.title}<span class="v">${fmt(last)}</span></div><div class="u">${c.unit} · spec ${c.spec}</div>`;line(card,c);grid.appendChild(card)}}
const CL=['var(--s1)','var(--s2)','var(--s3)'];
const sg=document.getElementById('scatter');
for(const f of D.scatter){const W=240,H=240,m=18;const card=document.createElement('div');card.className='card';card.innerHTML=`<div class="t">day ${Math.round(f.day)}<span class="v">${f.n} adults</span></div>`;
 const s=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'ideology scatter day '+f.day});const X=v=>m+(v+1)/2*(W-2*m),Y=v=>H-m-(v+1)/2*(H-2*m);
 el('rect',{x:m,y:m,width:W-2*m,height:H-2*m,fill:'none',stroke:'var(--grid)'},s);el('line',{x1:X(0),x2:X(0),y1:m,y2:H-m,stroke:'var(--grid)'},s);el('line',{x1:m,x2:W-m,y1:Y(0),y2:Y(0),stroke:'var(--grid)'},s);
 for(let i=0;i<f.x.length;i++){const c=el('circle',{cx:X(f.x[i]),cy:Y(f.y[i]),r:3,fill:CL[f.c[i]]||CL[0],'fill-opacity':.75,stroke:'var(--surface)','stroke-width':.75},s);
  c.addEventListener('mousemove',ev=>showTip(`axis1 ${f.x[i].toFixed(2)}, axis2 ${f.y[i].toFixed(2)}`,ev));c.addEventListener('mouseleave',hideTip)}
 card.appendChild(s);sg.appendChild(card)}
(function(){const host=document.getElementById('ridge');if(!D.hist.length){host.textContent='no panel data';return}
 const W=1100,rowH=34,H=D.hist.length*rowH+24,m=36;const s=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'ideology distribution over time'});const bins=D.hist[0].h.length;
 const mx=Math.max(...D.hist.flatMap(h=>h.h));const X=i=>m+i/(bins-1)*(W-2*m);
 D.hist.forEach((h,k)=>{const base=(k+1)*rowH;let d=`M${X(0)} ${base}`;h.h.forEach((v,i)=>{d+=`L${X(i).toFixed(1)} ${(base-v/mx*(rowH*1.6)).toFixed(1)}`});d+=`L${X(bins-1)} ${base}Z`;
  const p=el('path',{d,fill:'var(--s1)','fill-opacity':.35,stroke:'var(--s1)','stroke-width':1.5,'stroke-linejoin':'round'},s);
  const t=el('text',{x:m-6,y:base,'text-anchor':'end'},s);t.textContent='day '+Math.round(h.day);
  p.addEventListener('mousemove',ev=>showTip('day '+Math.round(h.day)+' · peak bin '+(h.h.indexOf(Math.max(...h.h))/(bins-1)*2-1).toFixed(2),ev));p.addEventListener('mouseleave',hideTip)});
 const t=el('text',{x:m,y:H-4},s);t.textContent='-1 (axis 1)';const t2=el('text',{x:W-m,y:H-4,'text-anchor':'end'},s);t2.textContent='+1';host.appendChild(s)})();
function bars(id,pair,labelFn){const host=document.getElementById(id);const sets=pair.filter(Boolean);if(!sets.length){host.textContent='no panel data';return}
 const keys=[...new Set(sets.flatMap(p=>p.rows.map(r=>r[0])))].sort((a,b)=>a-b);const mx=Math.max(...sets.flatMap(p=>p.rows.map(r=>r[1])));
 const W=320,H=130,m={l:26,b:16,t:6};const bw=(W-m.l)/keys.length;const s=el('svg',{viewBox:`0 0 ${W} ${H}`});
 keys.forEach((k,i)=>{sets.forEach((p,j)=>{const r=p.rows.find(r=>r[0]===k);const v=r?r[1]:0;const h=v/mx*(H-m.b-m.t);const w=bw/2-2;
  const b=el('rect',{x:m.l+i*bw+j*(w+1)+1,y:H-m.b-h,width:w,height:Math.max(h,0),rx:2,fill:j===0?'var(--muted)':'var(--s1)'},s);
  b.addEventListener('mousemove',ev=>showTip(`${labelFn(k)}: ${v} (day ${Math.round(p.day)})`,ev));b.addEventListener('mouseleave',hideTip)});
  if(keys.length<=14||i%2===0){const t=el('text',{x:m.l+i*bw+bw/2,y:H-3,'text-anchor':'middle'},s);t.textContent=k}});
 el('line',{x1:m.l,x2:W,y1:H-m.b,y2:H-m.b,stroke:'var(--axis)'},s);const t=el('text',{x:m.l-4,y:m.t+8,'text-anchor':'end'},s);t.textContent=mx;host.appendChild(s)}
bars('age',D.age,k=>`${k}–${k+9} years`);bars('deg',D.deg,k=>`degree ${k}`);
</script></main></body></html>
"""
