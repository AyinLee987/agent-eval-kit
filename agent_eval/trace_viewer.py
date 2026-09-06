"""Portable trace inspection without a server, account, or runtime dependency."""
from __future__ import annotations

import json
from pathlib import Path
import re

from .tracing import read_trace, sanitize_trace_value


def load_traces(source):
    source = Path(source)
    if source.is_file():
        paths = [source]
    else:
        directory = source / "traces" if (source / "traces").is_dir() else source
        paths = set(directory.glob("*.jsonl"))
        paths.update(path.with_name(path.name.removesuffix(".manifest.json") + ".jsonl")
                     for path in directory.glob("*.manifest.json"))
    records = {}
    report_path = source / "report.json" if source.is_dir() else source.parent.parent / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for record in report.get("records", []):
            references = [(record.get("trace", {}), False)]
            references.extend((attempt, True) for attempt in record.get("trace_attempts", []))
            for reference, historical in references:
                trace_id = reference.get("trace_id")
                if not trace_id:
                    continue
                keys = ("task_id", "condition", "trial_id") if historical else (
                    "task_id", "condition", "trial_id", "scores", "metric_statuses", "execution_error")
                records[trace_id] = {key: record.get(key) for key in keys}
                records[trace_id]["historical_attempt"] = historical
                if source.is_dir() and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", trace_id):
                    paths.add(directory / (trace_id + ".jsonl"))
    if not paths:
        raise ValueError("No JSONL traces found")
    traces = []
    for path in sorted(paths):
        trace = read_trace(path)
        if trace.get("trace_id") is None:
            trace["trace_id"] = path.stem
        trace["source_name"] = path.name
        trace["evaluation"] = records.get(trace.get("trace_id"))
        traces.append(trace)
    return traces


def export_trace_html(source, destination):
    destination = Path(destination)
    traces = load_traces(source)
    # Trace content is untrusted. Prevent a model's </script> from escaping the
    # inert JSON element; the viewer renders every field with textContent.
    payload = json.dumps(sanitize_trace_value(traces), ensure_ascii=False, allow_nan=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_HTML.replace("__TRACE_DATA__", payload), encoding="utf-8")
    return destination


_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; img-src data:; base-uri 'none'; form-action 'none'">
<title>Agent 评估轨迹</title><style>
:root{color-scheme:light;--ink:#162c35;--muted:#61727a;--line:#dce4e5;--accent:#087f76;--bg:#f5f7f6}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}button,input{font:inherit}header{padding:24px 32px;background:#fff;border-bottom:1px solid var(--line)}h1{margin:0;font-size:24px;letter-spacing:-.4px}header p{color:var(--muted);margin:5px 0 0}.eyebrow{color:var(--accent);font-size:11px;font-weight:750;letter-spacing:1.6px}.layout{display:grid;grid-template-columns:290px minmax(0,1fr);min-height:80vh}aside{padding:22px 16px;border-right:1px solid var(--line)}input{width:100%;padding:10px 12px;border:1px solid #cbd7d8;border-radius:8px;background:white;outline-color:var(--accent)}#count{padding:12px 4px;color:var(--muted);font-size:12px}.trace-button{width:100%;text-align:left;border:1px solid transparent;background:transparent;border-radius:9px;padding:12px;cursor:pointer;margin-bottom:6px;color:var(--ink);overflow-wrap:anywhere}.trace-button:hover{background:#eaf1ef}.trace-button.selected{background:#fff;border-color:#97c9c1;box-shadow:0 2px 7px #152c3510}.trace-button b{display:block}.trace-button small{color:var(--muted);display:block}main{padding:26px 30px;min-width:0}h2{font-size:21px;margin:0 0 4px}.subtle{color:var(--muted);font-size:12px;overflow-wrap:anywhere}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:20px 0}.metric{padding:13px 17px;background:#fff;border:1px solid var(--line);border-radius:10px}.metric strong{display:block;font-size:23px;font-weight:650}.metric span{color:var(--muted);font-size:12px}.notice{border-left:3px solid #df9b34;padding:10px 14px;background:#fff7e9;margin:12px 0;overflow-wrap:anywhere}.panel{background:#fff;border:1px solid var(--line);border-radius:10px;margin:14px 0;overflow:hidden}.panel-title{padding:12px 17px;border-bottom:1px solid var(--line);font-weight:650}details{min-width:0}summary{cursor:pointer;list-style:none}summary::-webkit-details-marker{display:none}.span>summary{display:grid;grid-template-columns:minmax(120px,1fr) 88px 180px 88px;align-items:center;gap:12px;padding:12px 16px 12px calc(16px + var(--depth,0)*18px);border-bottom:1px solid #e9efef}.span>summary:hover{background:#f8faf9}.span-name{overflow-wrap:anywhere}.span-name:before{content:'▸';color:var(--muted);margin-right:9px}.span[open]>.span-name:before,.span[open]>summary .span-name:before{content:'▾'}.badge{font-size:11px;padding:2px 8px;background:#e8f5ee;color:#216441;border-radius:5px;text-align:center;overflow-wrap:anywhere}.badge.error{color:#ab3440;background:#ffedf0}.badge.pending{background:#fff2d8;color:#936010}.kind{color:var(--muted);font-size:11px}.bar-track{height:7px;background:#edf2f1;border-radius:10px;overflow:hidden}.bar{height:100%;background:#52aea0;border-radius:10px;min-width:2px}.duration{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted);font-size:12px}.body{padding:12px 18px 18px;background:#fbfcfc}.body label{display:block;font-weight:650;font-size:12px;margin:10px 0 5px}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;font:12px/1.65 ui-monospace,Consolas,monospace;padding:12px;background:#f0f4f4;border:1px solid #e1e8e8;border-radius:7px;max-height:420px;overflow:auto}.raw>summary{padding:13px 17px;font-weight:650}.raw pre{margin:0 16px 16px}.empty{padding:24px;color:var(--muted)}footer{margin-top:18px;color:var(--muted);font-size:12px}@media(max-width:900px){.layout{grid-template-columns:220px minmax(0,1fr)}main{padding:20px 16px}.span>summary{grid-template-columns:minmax(120px,1fr) 72px 65px}.bar-track{display:none}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){header{padding:20px}.layout{display:block}aside{max-height:270px;overflow:auto;border-bottom:1px solid var(--line)}.span>summary{padding-left:calc(10px + var(--depth,0)*8px)}}
</style></head><body><header><div class="eyebrow">AGENT EVAL / LOCAL OBSERVABILITY</div><h1>Agent 评估轨迹</h1><p>展开模型与工具调用，查看输入、输出、错误和时间。所有数据保存在这个页面中。</p></header>
<div class="layout"><aside><input id="search" aria-label="搜索轨迹" placeholder="搜索用例、条件或 Trace ID"><div id="count"></div><nav id="trace-list" aria-label="轨迹列表"></nav></aside><main id="detail"></main></div>
<script id="trace-data" type="application/json">__TRACE_DATA__</script><script>
'use strict';const traces=JSON.parse(document.getElementById('trace-data').textContent);let selected=0;
function el(tag,cls,text){const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=String(text);return node}
function json(value){return JSON.stringify(value,null,2)}
function first(trace){return trace.events.find(e=>e.event==='trace.start')||trace.events[0]||trace.manifest||trace.evaluation||{}}
function ms(value){return Number.isFinite(value)?(value<1000?value.toFixed(1)+' ms':(value/1000).toFixed(2)+' s'):'未完成'}
function startOf(s){return s.start||s.start_event||s}function endOf(s){return s.end||s.end_event||{}}
function duration(s){const a=startOf(s),b=endOf(s);if(Number.isFinite(b.duration_ms))return b.duration_ms;if(Number.isFinite(s.duration_ms))return s.duration_ms;return Number.isFinite(b.monotonic_ns)&&Number.isFinite(a.monotonic_ns)?(b.monotonic_ns-a.monotonic_ns)/1e6:null}
function list(){const box=document.getElementById('trace-list');box.replaceChildren();let count=0;const q=document.getElementById('search').value.toLowerCase();traces.forEach((trace,i)=>{const f=first(trace);if(!json([trace.trace_id,f.task_id,f.condition]).toLowerCase().includes(q))return;count++;const b=el('button','trace-button'+(selected===i?' selected':''));b.type='button';b.append(el('b','',f.task_id||trace.source_name),el('small','',(f.condition||'default')+' · trial '+(f.trial_id??0)),el('small','',trace.complete?'记录完整':'记录不完整'));b.onclick=()=>{selected=i;list();detail()};box.append(b)});document.getElementById('count').textContent=count+' / '+traces.length+' 条轨迹'}
function detail(){const trace=traces[selected],f=first(trace),main=document.getElementById('detail');main.replaceChildren(el('h2','',f.task_id||'Trace'),el('div','subtle',(f.condition||'default')+' · trial '+(f.trial_id??0)+' · '+(trace.trace_id||'unknown')));const spans=Array.isArray(trace.spans)?trace.spans:Object.values(trace.spans||{});const starts=spans.map(startOf),ends=spans.map(endOf);const times=trace.events.map(e=>e.monotonic_ns).filter(Number.isFinite);const elapsed=times.length?(Math.max(...times)-Math.min(...times))/1e6:null;let tokens=0,known=0;spans.forEach(s=>{const a=startOf(s),b=endOf(s);if(a.kind!=='llm')return;const u=b.usage||s.usage||{};const n=u.total_tokens??((Number.isFinite(u.prompt_tokens)&&Number.isFinite(u.completion_tokens))?u.prompt_tokens+u.completion_tokens:null);if(Number.isFinite(n)){tokens+=n;known++}});const metrics=el('div','metrics');for(const [value,label]of [[ms(elapsed),'记录时间范围'],[spans.length,'Span 数'],[starts.filter(s=>s.kind==='llm').length,'模型调用'],[known?tokens.toLocaleString():'未提供','已记录模型 Token']]){const m=el('div','metric');m.append(el('strong','',value),el('span','',label));metrics.append(m)}main.append(metrics);
if(!trace.complete||trace.diagnostics?.length){const n=el('div','notice');n.append(el('b','',trace.complete?'记录诊断':'轨迹不完整：缺失部分不能视为成功'),el('pre','',json(trace.diagnostics||[])));main.append(n)}
const panel=el('section','panel');panel.append(el('div','panel-title','调用树与时间线'));const byId=new Map(spans.map(s=>[startOf(s).span_id||s.span_id,s]));const seen=new Set();const origin=times.length?Math.min(...times):0;function draw(s,depth){const a=startOf(s),b=endOf(s),id=a.span_id||s.span_id;if(seen.has(s))return;seen.add(s);const d=el('details','span');d.style.setProperty('--depth',Math.min(depth,12));const row=el('summary');const name=el('div','span-name');name.append(document.createTextNode(a.name||s.name||id||'span'),el('div','kind',a.kind||s.kind||''));const status=b.status||s.status||'pending';const badge=el('span','badge'+(['error','failed','cancelled'].includes(status)?' error':status==='pending'?' pending':''),status);const track=el('div','bar-track');const bar=el('div','bar');const delta=duration(s);bar.style.marginLeft=(elapsed?Math.min(99,Math.max(0,((a.monotonic_ns-origin)/1e6/elapsed)*100)):0)+'%';bar.style.width=(elapsed&&delta!==null?Math.min(100,Math.max(1,delta/elapsed*100)):1)+'%';track.append(bar);row.append(name,badge,track,el('span','duration',ms(delta)));d.append(row);const body=el('div','body');for(const [label,value]of [['输入',a.input],['输出',b.output],['错误',b.error],['Token 用量',b.usage],['开始元数据',a.metadata],['结束元数据',b.metadata]])if(value!==undefined&&value!==null){body.append(el('label','',label),el('pre','',json(value)))}const events=trace.events.filter(e=>e.span_id===id&&e.event!=='span.start'&&e.event!=='span.end');if(events.length)body.append(el('label','','事件'),el('pre','',json(events)));body.append(el('div','subtle','span_id: '+id+' · parent: '+(a.parent_span_id||'root')));d.append(body);panel.append(d);spans.filter(c=>startOf(c).parent_span_id===id).forEach(c=>draw(c,depth+1))}
spans.filter(s=>!byId.has(startOf(s).parent_span_id)).forEach(s=>draw(s,0));spans.forEach(s=>draw(s,0));if(!spans.length)panel.append(el('div','empty','未记录调用 Span；请检查中断或埋点配置。'));main.append(panel);for(const [label,value]of [['评估分数与关联',trace.evaluation],['完整事件流',trace.events]])if(value){const d=el('details','panel raw');d.append(el('summary','',label),el('pre','',json(value)));main.append(d)}main.append(el('footer','','记录完整表示事件已闭合，不代表任务通过；Token 仅汇总提供用量的模型调用。未接入埋点的外部进程不可见。'))}
document.getElementById('search').addEventListener('input',list);list();detail();
</script></body></html>'''
