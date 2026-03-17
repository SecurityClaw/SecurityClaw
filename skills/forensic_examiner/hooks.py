from __future__ import annotations

import re


def build_followup_question(result: dict) -> str:
    report = result.get("forensic_report", {}) or {}
    incident = report.get("incident_summary") or "the investigated activity"
    anchors = report.get("context_anchors") or {}

    entities = []
    if anchors.get("ips"):
        entities.append("IPs: " + ", ".join(str(ip) for ip in anchors["ips"][:8]))
    if anchors.get("ports"):
        entities.append("Ports: " + ", ".join(str(port) for port in anchors["ports"][:8]))
    if anchors.get("countries"):
        entities.append("Countries: " + ", ".join(str(country) for country in anchors["countries"][:6]))
    if anchors.get("protocols"):
        entities.append("Protocols: " + ", ".join(str(protocol) for protocol in anchors["protocols"][:6]))

    if not entities:
        report_text = "\n".join(str(value) for value in report.values() if isinstance(value, (str, int, float)))
        ips = list(dict.fromkeys(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", report_text)))
        ports = list(dict.fromkeys(re.findall(r"\bport\s*(\d{1,5})\b", report_text, flags=re.IGNORECASE)))
        if ips:
            entities.append("IPs: " + ", ".join(ips[:8]))
        if ports:
            entities.append("Ports: " + ", ".join(ports[:8]))

    if not entities:
        return ""

    return (
        f"Assess the reputation and threat context for entities identified during this forensic investigation: {incident}. "
        + " ".join(entities)
    )


def post_success(result: dict, manifests: dict[str, dict], executed_skills: list[str], user_question: str, **_: object) -> list[dict]:
    question = build_followup_question(result)
    if not question:
        return []

    threat_skill = None
    for skill_name, manifest in manifests.items():
        if manifest.get("routing_group") == "threat_intel":
            threat_skill = skill_name
            break

    if not threat_skill or threat_skill in executed_skills:
        return []

    return [{
        "skill": threat_skill,
        "parameters": {"question": question},
    }]


def format_response(user_question: str, result: dict, skill_results: dict | None = None) -> str:
    report = result.get("forensic_report", {}) or {}
    incident = report.get("incident_summary") or user_question
    results_found = report.get("results_found", 0)
    refinements = report.get("refinement_rounds", 0)
    narrative = report.get("timeline_narrative", "") or ""

    timeline_lines = []
    for line in narrative.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}:\d{2}\b|\bUTC\b", line_stripped, re.IGNORECASE):
            timeline_lines.append(line_stripped)
        if len(timeline_lines) >= 6:
            break

    if not timeline_lines and narrative:
        timeline_lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", narrative) if s.strip()][:4]

    entities = sorted(set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", incident + "\n" + narrative)))
    ports = sorted(set(re.findall(r"\bport\s*(\d{1,5})\b|\b(\d{2,5})/tcp\b|\b(\d{2,5})/udp\b", (incident + "\n" + narrative).lower())))
    flat_ports = sorted({p for tup in ports for p in tup if p})

    paragraph1 = (
        f"Forensic report for '{incident}': analyzed {results_found} matching events "
        f"with {refinements} refinement rounds. The objective was incident reconstruction "
        "with timeline, behavior frequency, key entities, and risk implications."
    )
    timeline_text = "\n".join([f"- {line}" for line in timeline_lines]) if timeline_lines else "- No concrete timestamped events were returned by the data source."
    paragraph2 = f"Timeline:\n{timeline_text}"

    entity_text = ", ".join(entities[:10]) if entities else "No IP entities extracted"
    port_text = ", ".join(flat_ports[:10]) if flat_ports else "No explicit ports extracted"
    pattern_hint = ""
    for sentence in re.split(r"(?<=[.!?])\s+", narrative):
        if re.search(r"pattern|periodic|sporadic|frequency|interval|automated|bot|risk|threat", sentence, re.IGNORECASE):
            pattern_hint = sentence.strip()
            break
    if not pattern_hint:
        pattern_hint = "Pattern/risk signal was not explicit in the model output and should be treated as low confidence."
    paragraph3 = (
        f"Entities and behavior: IPs involved: {entity_text}. Ports involved: {port_text}. "
        f"Frequency/pattern assessment: {pattern_hint}"
    )

    threat_result = (skill_results or {}).get("threat_analyst") or {}
    if threat_result.get("status") == "ok" and threat_result.get("verdicts"):
        verdict_lines = []
        for verdict in threat_result.get("verdicts", [])[:3]:
            verdict_lines.append(
                f"- {verdict.get('verdict', 'UNKNOWN')} ({verdict.get('confidence', 0)}% confidence): {(verdict.get('reasoning') or '').strip().replace(chr(10), ' ')}"
            )
        paragraph4 = "Reputation and threat intel:\n" + "\n".join(verdict_lines)
    else:
        paragraph4 = (
            "Reputation and threat intel: no explicit reputation verdict was returned. "
            "If API keys are configured, rerun with threat_analyst enabled to include AbuseIPDB/VirusTotal/OTX/Talos signals."
        )

    return f"{paragraph1}\n\n{paragraph2}\n\n{paragraph3}\n\n{paragraph4}"


def build_threat_followup_question(forensic_result: dict) -> str:
    """Build a question for auto-chained threat_analyst from forensic results.
    
    This hook is called after forensic_examiner completes successfully to create
    a follow-up question for threat_analyst analysis of entities discovered in
    the forensic timeline.
    
    Args:
        forensic_result: Result dict from forensic_examiner skill
    
    Returns:
        Question string for threat_analyst, or empty string if no following needed
    """
    report = forensic_result.get("forensic_report", {}) if forensic_result else {}
    incident = report.get("incident_summary", "")
    timeline = (report.get("timeline_narrative", "") or "")[:800]
    anchors = report.get("context_anchors", {}) or {}
    ips = anchors.get("ips", [])[:5]
    ports = anchors.get("ports", [])[:3]
    countries = anchors.get("countries", [])[:3]
    protocols = anchors.get("protocols", [])[:3]
    
    anchor_text = (
        f"Anchors: IPs={ips}, Ports={ports}, Countries={countries}, Protocols={protocols}."
        if (ips or ports or countries or protocols)
        else ""
    )

    if not incident and not timeline:
        return ""

    return (
        "Perform threat reputation analysis for entities in this forensic report. "
        "Prioritize the provided anchor entities and do not pivot to unrelated IPs unless strongly justified by evidence. "
        "Focus on maliciousness signals, confidence, and actionable response. "
        f"Incident: {incident}\n"
        f"{anchor_text}\n"
        f"Timeline excerpt: {timeline}"
    )

    return "\n\n".join([paragraph1, paragraph2, paragraph3, paragraph4])