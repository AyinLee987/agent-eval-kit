"""Knowledge base for the medical-QA benchmark — adapted from real,
publicly-reviewed sources, not synthetic and not the author's own medical
knowledge.

Every document is a Chinese adaptation (translated and reorganized, not a
verbatim copy) of a specific MedlinePlus Medical Encyclopedia page —
MedlinePlus is produced by the US National Library of Medicine (NIH); its
encyclopedia content is medically reviewed and, per NIH policy, US
government works are in the public domain. ``source_url`` on every
document is the exact page adapted, preserved end-to-end through ingestion
(``RAGIngestionService.ingest_text(source_url=...)``) so it surfaces in
every retrieved citation (``format_evidence_context``) — this is what lets
the "clinical_accuracy" judge dimension check whether a claim traces to a
real, checkable source instead of being invented.

Scope is deliberately narrow: five common, self-limiting, low-risk
conditions (never a substitute for the "predict a diagnosis for anything"
scope creep a real product would need much more care for), plus one
consolidated red-flag/emergency-warning-signs document that exists purely
as an escalation trigger — it is never a source for a tentative diagnosis,
only for "this doesn't match the low-risk documents, escalate instead."

None of this is a real clinical guideline and none of it should be treated
as medical advice — it exists to give a benchmark agent something to
retrieve and cite, not to inform an actual care decision.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, TypedDict


class Document(TypedDict):
    logical_id: str
    title: str
    source_url: str
    sections: List[Tuple[str, str]]


DOCUMENTS: List[Document] = [
    {
        "logical_id": "common-cold",
        "title": "普通感冒",
        "source_url": "https://medlineplus.gov/ency/article/000678.htm",
        "sections": [
            (
                "定义与病因",
                "普通感冒是一种病毒性上呼吸道感染，主要表现为鼻部症状和轻度全身不适。"
                "感冒通过感染者打喷嚏、咳嗽或擤鼻涕时产生的飞沫传播，也可通过触摸被污染的"
                "物体表面（如玩具、门把手）后再触摸口鼻传播。患者在感染后最初2到3天传染性最强。",
            ),
            (
                "典型症状",
                "主要症状包括鼻塞、流涕、咽部发干发痒、打喷嚏。其他可能出现的症状包括咳嗽、"
                "食欲下降、头痛、肌肉酸痛、鼻后滴漏、咽痛。儿童感冒时常伴有37.8-38.9摄氏度左右"
                "的低热，成人发热则较少见或程度轻微。",
            ),
            (
                "自我护理建议",
                "充分休息并多饮水。非处方感冒药可以缓解症状，但不会加快康复速度。感冒是病毒"
                "感染，抗生素对其无效，不应自行使用抗生素。使用中草药或膳食补充剂前应先咨询"
                "医生。",
            ),
            (
                "何时需要就医",
                "如果出现呼吸困难，应联系医生。如果症状在7到10天后仍未改善甚至加重，也应联系"
                "医生。",
            ),
        ],
    },
    {
        "logical_id": "allergic-rhinitis",
        "title": "过敏性鼻炎（花粉症）",
        "source_url": "https://medlineplus.gov/ency/patientinstructions/000547.htm",
        "sections": [
            (
                "定义",
                "过敏性鼻炎是接触过敏原（如尘螨、动物皮屑、花粉）后出现的一组鼻部症状，"
                "俗称花粉症。",
            ),
            (
                "典型症状",
                "常见症状是流清水样鼻涕和鼻痒，也可能伴有眼部不适（眼痒、流泪）。这些症状"
                "通常没有发热。",
            ),
            (
                "自我护理建议",
                "减少接触过敏原是基础：控制尘螨、处理室内外霉菌、减少花粉和宠物毛发接触，"
                "可使用空气过滤设备、去除地毯、使用除湿机。鼻用糖皮质激素喷雾是最有效的治疗"
                "方式，坚持每日使用效果最好；抗组胺药适合偶发症状，较新的品种嗜睡副作用较小；"
                "减充血剂可缓解鼻塞，但作为鼻喷剂使用不应连续超过3天；生理盐水鼻腔冲洗对轻症"
                "也有帮助。",
            ),
            (
                "何时需要就医",
                "如果症状严重、经上述自我护理和药物治疗后仍不改善，或出现喘息、咳嗽加重，"
                "应联系医生。",
            ),
        ],
    },
    {
        "logical_id": "tension-headache",
        "title": "紧张性头痛",
        "source_url": "https://www.medlineplus.gov/ency/article/000797.htm",
        "sections": [
            (
                "定义",
                "紧张性头痛是最常见的头痛类型，表现为头部、头皮或颈部的疼痛或不适，通常"
                "伴随这些部位的肌肉紧张。",
            ),
            (
                "典型症状",
                "疼痛通常是钝痛、压迫性的（不是搏动性的），常被描述为像有一条紧箍带或虎钳"
                "箍在头部周围，往往累及整个头部而非局限于一点或一侧，可能在头皮、太阳穴或"
                "后颈部、肩部更明显。一次发作可持续30分钟到7天不等。",
            ),
            (
                "自我护理建议",
                "记录头痛日记以找出诱因；尝试按摩、生物反馈等减压方法；冷敷或热敷；保证"
                "充分休息；保持良好姿势，使用电脑时定期做颈部活动，尝试更换枕头。",
            ),
            (
                "何时需要立即就医（拨打急救电话或前往急诊）",
                "出现以下情况应立即拨打急救电话或前往急诊：这是一生中最严重的一次头痛，"
                "且影响到了日常活动；举重、有氧运动、慢跑或性行为后立即出现头痛；头痛突然"
                "发作、呈爆炸性或剧烈性质；头痛是“有史以来最严重的”，即使平时也经常头痛；"
                "伴随言语不清、视力改变、肢体活动障碍、平衡问题、意识混乱或记忆丧失等神经系统"
                "症状；头痛在24小时内持续加重；伴随发热、颈部僵硬、恶心呕吐；头部外伤后出现"
                "头痛；单眼剧痛并伴该侧眼睛发红；50岁以上人群新发头痛；头痛伴视觉问题和咀嚼"
                "时疼痛，或伴体重下降；有癌症病史者新发头痛；艾滋病、化疗中或长期使用类固醇"
                "等免疫功能低下人群出现头痛。",
            ),
            (
                "何时需要一般就医（非紧急）",
                "如果头痛会把人从睡眠中痛醒、持续数天不缓解、早晨明显加重、头痛的模式或"
                "强度发生变化，或在没有明确原因的情况下频繁发作，应尽快预约医生就诊，但不"
                "属于需要立即拨打急救电话的情况。",
            ),
        ],
    },
    {
        "logical_id": "indigestion",
        "title": "消化不良",
        "source_url": "https://medlineplus.gov/ency/article/003260.htm",
        "sections": [
            (
                "定义",
                "消化不良（医学上称为dyspepsia）是上腹部的轻度不适，通常在进餐过程中或"
                "餐后不久出现。消化不良与烧心（胃酸反流引起的胸骨后灼烧感）是不同的概念。",
            ),
            (
                "典型症状",
                "常见表现为肚脐与胸骨下端之间区域的灼热感、烧灼感或疼痛感，以及进餐开始"
                "后不久或结束时出现的不适饱胀感；腹胀和恶心相对少见。",
            ),
            (
                "自我护理建议",
                "从容进餐，避免进餐时争吵或情绪激动；餐后避免剧烈活动或运动；细嚼慢咽；"
                "压力大时尝试放松练习；避免阿司匹林、非甾体抗炎药、酒精和吸烟；可以考虑"
                "非处方抗酸药物（如法莫替丁、奥美拉唑等，具体用药请遵医嘱或说明书）。",
            ),
            (
                "何时需要立即就医",
                "如果消化不良同时伴有下颌疼痛、胸痛、背痛、大量出汗、焦虑或有大祸临头的"
                "感觉，这些可能是心脏病发作的表现，应立即就医，不要拖延。",
            ),
            (
                "何时需要一般就医",
                "如果症状明显改变、持续数天没有缓解，或出现不明原因体重下降、突发剧烈"
                "腹痛、吞咽困难、皮肤或眼睛发黄、呕血或黑便，应联系医生。",
            ),
        ],
    },
    {
        "logical_id": "muscle-strain",
        "title": "轻度肌肉拉伤",
        "source_url": "https://medlineplus.gov/ency/article/000042.htm",
        "sections": [
            (
                "定义",
                "肌肉拉伤是指肌肉被过度牵拉导致部分纤维撕裂，俗称拉伤。常见诱因包括运动"
                "量过大、运动前热身不充分、柔韧性不足或跌倒等意外。",
            ),
            (
                "典型症状",
                "受伤部位疼痛、活动受限，可能出现皮肤淤青变色和局部肿胀。",
            ),
            (
                "自我护理与急救建议",
                "受伤后应立即冰敷（用布包裹冰袋，不要让冰直接接触皮肤），第一天每小时"
                "冰敷10到15分钟，之后每3到4小时一次；前三天以冰敷为主，之后如果仍然疼痛，"
                "冷敷热敷交替也可以；让受伤肌肉至少休息一天，条件允许时将患处抬高至高于"
                "心脏位置；疼痛期间避免使用受伤肌肉，随疼痛减轻再逐渐增加活动量并做轻柔"
                "拉伸。",
            ),
            (
                "何时需要立即就医",
                "如果完全无法移动受伤的肌肉，或伤处有出血，应立即拨打急救电话或前往急诊。",
            ),
            (
                "何时需要一般就医",
                "如果按上述方法处理后，疼痛在数周后仍未缓解，应联系医生。",
            ),
        ],
    },
    {
        "logical_id": "emergency-warning-signs",
        "title": "紧急就医预警清单",
        "source_url": "https://medlineplus.gov/ency/patientinstructions/000593.htm",
        "sections": [
            (
                "说明",
                "本文档不对应任何具体诊断，只用于判断某个情况是否需要立即就医——只要用户"
                "描述的情况匹配下列任意一条，就应当建议立即拨打急救电话或前往急诊，而不是"
                "针对该情况给出推测性判断或自我护理建议。",
            ),
            (
                "应立即拨打急救电话，不要等待",
                "出现以下情况应立即拨打急救电话：窒息；呼吸停止；头部受伤并伴有晕厥、"
                "意识丧失或意识混乱；颈部或脊柱受伤，尤其伴有肢体麻木或无法活动；"
                "触电或被雷击；严重烧伤；严重胸痛或胸部压迫感；严重呼吸困难；癫痫发作持续"
                "超过1分钟，或发作后未能迅速清醒；突然无法说话、看东西、行走或活动；身体"
                "一侧突然无力或下垂。",
            ),
            (
                "应前往急诊或拨打急救电话",
                "出现以下情况应前往急诊或拨打急救电话：呼吸困难；晕厥；手臂或下颌疼痛；"
                "突发的或异常剧烈的头痛；持续不消退的头晕或虚弱；吸入烟雾或有毒气体；突然"
                "意识混乱；大量出血；可能骨折并伴有活动障碍，尤其是骨头穿出皮肤；较深的伤口；"
                "严重烧伤；咳血或呕血；身体任何部位的剧烈疼痛；严重过敏反应（伴呼吸困难、"
                "肿胀、荨麻疹）；发热伴头痛和颈部僵硬；药物无法缓解的高烧；持续呕吐或腹泻；"
                "药物、毒品或酒精中毒或过量；癫痫发作；出现自杀念头（可拨打心理援助热线或"
                "前往急诊）。",
            ),
        ],
    },
]


def render_markdown(document: Document) -> str:
    """Render a document's sections as ``## heading`` Markdown blocks."""

    return "\n\n".join(f"## {heading}\n{body}" for heading, body in document["sections"])


def all_facts() -> Dict[str, str]:
    """Map every ``"logical_id::heading"`` to its body text, for sanity checks."""

    return {
        f"{doc['logical_id']}::{heading}": body
        for doc in DOCUMENTS
        for heading, body in doc["sections"]
    }
