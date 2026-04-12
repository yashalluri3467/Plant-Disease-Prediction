from typing import Dict, List, Tuple


_REMEDY_LIBRARY: Dict[str, Dict[str, List[str] | str]] = {
    "healthy": {
        "summary": "The plant appears healthy. Continue good crop management to keep disease pressure low.",
        "actions": [
            "Keep regular irrigation and avoid water stress.",
            "Apply balanced nutrients based on crop stage.",
            "Inspect leaves weekly for early symptoms.",
        ],
        "prevention": [
            "Use clean tools and sanitize pruning blades.",
            "Avoid overcrowding to improve airflow.",
        ],
    },
    "powdery_mildew": {
        "summary": "Powdery mildew is a fungal disease that spreads quickly in humid and poorly ventilated conditions.",
        "actions": [
            "Remove heavily infected leaves and destroy them away from the field.",
            "Spray sulfur- or potassium bicarbonate-based fungicide as per product label.",
            "Avoid overhead watering late in the day.",
        ],
        "prevention": [
            "Maintain spacing for airflow.",
            "Use resistant varieties when available.",
        ],
    },
    "early_blight": {
        "summary": "Early blight is a common fungal disease causing dark concentric spots on older leaves.",
        "actions": [
            "Remove infected lower leaves first.",
            "Use registered fungicide (for example chlorothalonil or mancozeb) as per local guidance.",
            "Mulch around plants to reduce soil splash onto leaves.",
        ],
        "prevention": [
            "Follow crop rotation of at least 2 years.",
            "Avoid wetting foliage during irrigation.",
        ],
    },
    "late_blight": {
        "summary": "Late blight is aggressive and can spread rapidly under cool and wet conditions.",
        "actions": [
            "Isolate infected plants immediately.",
            "Apply a suitable anti-oomycete fungicide according to local agricultural recommendations.",
            "Destroy severely infected plants to limit spread.",
        ],
        "prevention": [
            "Avoid dense canopy and improve drainage.",
            "Monitor weather and spray preventively during high-risk periods.",
        ],
    },
    "leaf_spot": {
        "summary": "Leaf spot diseases can be fungal or bacterial and usually spread through water splash and contact.",
        "actions": [
            "Prune and remove spotted leaves.",
            "Apply copper-based or other recommended protectant spray per label.",
            "Water near the root zone to keep leaves dry.",
        ],
        "prevention": [
            "Do not work with plants while leaves are wet.",
            "Rotate crops and clear plant debris after harvest.",
        ],
    },
    "rust": {
        "summary": "Rust diseases cause orange-brown pustules and can reduce photosynthesis significantly.",
        "actions": [
            "Remove severely affected leaves.",
            "Use an appropriate fungicide program for rust management.",
            "Increase sunlight exposure and airflow where possible.",
        ],
        "prevention": [
            "Avoid excessive nitrogen that drives soft growth.",
            "Use resistant cultivars if available.",
        ],
    },
    "blight": {
        "summary": "Blight symptoms indicate fast tissue death and require quick intervention.",
        "actions": [
            "Cut and dispose infected plant parts away from the field.",
            "Start a protective fungicide schedule as advised locally.",
            "Reduce moisture on foliage and improve ventilation.",
        ],
        "prevention": [
            "Use disease-free seed and clean transplant material.",
            "Maintain field hygiene and remove old crop residue.",
        ],
    },
    "mosaic_virus": {
        "summary": "Mosaic viruses often spread by insects and infected planting material.",
        "actions": [
            "Remove and destroy infected plants immediately.",
            "Control aphids/whiteflies with integrated pest management.",
            "Do not compost infected plants.",
        ],
        "prevention": [
            "Use virus-free seeds or seedlings.",
            "Control weed hosts around the field.",
        ],
    },
    "bacterial_spot": {
        "summary": "Bacterial spot thrives in warm wet weather and spreads through splash and tools.",
        "actions": [
            "Avoid touching plants when wet.",
            "Use copper-based bactericide where recommended.",
            "Remove heavily infected leaves and sanitize tools.",
        ],
        "prevention": [
            "Use clean seeds/transplants.",
            "Avoid overhead irrigation.",
        ],
    },
    "default": {
        "summary": "Disease class detected, but a specific remedy profile is not available yet.",
        "actions": [
            "Confirm diagnosis with local agricultural extension experts.",
            "Remove visibly infected plant tissue to reduce spread.",
            "Follow crop-specific fungicide/insecticide guidance from local authorities.",
        ],
        "prevention": [
            "Keep field and tools sanitized.",
            "Use resistant varieties and crop rotation where possible.",
        ],
    },
}


def _parse_label(label: str) -> Tuple[str, str]:
    if "___" in label:
        crop, disease = label.split("___", 1)
        return crop.replace("_", " "), disease.replace("_", " ")

    parts = label.replace("__", "_").split("_")
    if len(parts) > 1:
        return parts[0], " ".join(parts[1:])
    return "Unknown crop", label.replace("_", " ")


def _match_key(disease_name: str) -> str:
    value = disease_name.lower().strip()
    if "healthy" in value:
        return "healthy"
    if "powdery" in value and "mildew" in value:
        return "powdery_mildew"
    if "early" in value and "blight" in value:
        return "early_blight"
    if "late" in value and "blight" in value:
        return "late_blight"
    if "leaf" in value and "spot" in value:
        return "leaf_spot"
    if "bacterial" in value and "spot" in value:
        return "bacterial_spot"
    if "rust" in value:
        return "rust"
    if "mosaic" in value or "virus" in value:
        return "mosaic_virus"
    if "blight" in value:
        return "blight"
    return "default"


def get_local_remedy(label: str) -> Dict[str, object]:
    crop_name, disease_name = _parse_label(label)
    key = _match_key(disease_name)
    profile = _REMEDY_LIBRARY[key]

    return {
        "crop": crop_name.title(),
        "disease": disease_name.title(),
        "summary": profile["summary"],
        "actions": profile["actions"],
        "prevention": profile["prevention"],
        "safety_note": "Always follow local agricultural rules and product labels before applying chemicals.",
    }
