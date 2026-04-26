"""
build_skill_db.py  —  Corrected Version

Key fixes:
  - Replaced raw TF-IDF token extraction with a curated skill taxonomy
  - Skills are real, learnable skills (not stopwords like 'monthly', 'cash', 'office')
  - Each category has three tiers: high_priority, medium, tools
  - Saved as a rich dict so app.py can show prioritized recommendations
"""

import pickle
import os

os.makedirs("model", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CURATED SKILL TAXONOMY
# Each role has three tiers:
#   high     → core skills employers look for first
#   medium   → secondary skills that strengthen the profile
#   tools    → software / platforms relevant to the role
#
# Matching in app.py uses lowercase so keep all values lowercase.
# ─────────────────────────────────────────────────────────────
SKILL_TAXONOMY = {

    "ACCOUNTANT": {
        "high":   ["financial reporting", "tally", "gst filing", "audit", "taxation",
                   "balance sheet", "accounts payable", "accounts receivable"],
        "medium": ["budgeting", "forecasting", "payroll management", "cash flow analysis",
                   "profit and loss", "cost accounting", "bank reconciliation"],
        "tools":  ["sap fico", "quickbooks", "zoho books", "microsoft excel", "busy accounting"],
    },

    "ADVOCATE": {
        "high":   ["legal research", "case management", "contract drafting", "litigation",
                   "legal documentation", "court pleadings"],
        "medium": ["corporate law", "criminal law", "civil law", "intellectual property",
                   "labour law", "legal advisory"],
        "tools":  ["manupatra", "scc online", "lexisnexis", "ms word"],
    },

    "AGRICULTURE": {
        "high":   ["crop management", "soil testing", "irrigation management", "pest control",
                   "agronomy", "farm planning"],
        "medium": ["fertilizer application", "crop yield optimization", "horticulture",
                   "animal husbandry", "agricultural extension"],
        "tools":  ["gis mapping", "drone technology", "farm management software", "ndvi analysis"],
    },

    "APPAREL": {
        "high":   ["garment construction", "pattern making", "fabric selection", "quality control",
                   "production planning", "merchandising"],
        "medium": ["fashion design", "trend analysis", "textile knowledge", "cost estimation",
                   "supply chain", "retail buying"],
        "tools":  ["lectra", "adobe illustrator", "coreldraw", "erp systems"],
    },

    "ARTS": {
        "high":   ["fine arts", "creative direction", "visual composition", "art curation",
                   "illustration", "color theory"],
        "medium": ["art history", "exhibition planning", "creative writing", "photography",
                   "printmaking", "mixed media"],
        "tools":  ["adobe photoshop", "procreate", "lightroom", "sketchbook", "canva"],
    },

    "AUTOMOBILE": {
        "high":   ["engine diagnostics", "transmission repair", "brake systems",
                   "electrical systems", "vehicle maintenance", "suspension systems"],
        "medium": ["ev battery maintenance", "bodywork", "emission testing",
                   "wheel alignment", "workshop management"],
        "tools":  ["obd-ii scanner", "autocad", "dms software", "alldata"],
    },

    "AVIATION": {
        "high":   ["aircraft maintenance", "flight operations", "air traffic control",
                   "aviation safety", "navigation systems", "crew resource management"],
        "medium": ["ground operations", "cargo handling", "passenger services",
                   "weather analysis", "emergency procedures"],
        "tools":  ["amos", "sita", "sabre", "jeppesen"],
    },

    "BANKING": {
        "high":   ["credit analysis", "loan processing", "kyc compliance",
                   "treasury operations", "branch operations", "lending"],
        "medium": ["retail banking", "trade finance", "forex",
                   "anti money laundering", "regulatory compliance", "npa management"],
        "tools":  ["finacle", "sap banking", "core banking systems", "ms excel", "swift"],
    },

    "BPO": {
        "high":   ["customer support", "inbound calls", "outbound calls", "ticket resolution",
                   "voice process", "escalation management"],
        "medium": ["quality assurance", "upselling", "cross selling", "chat support",
                   "email support", "kpi tracking"],
        "tools":  ["crm software", "zendesk", "salesforce", "freshdesk", "avaya"],
    },

    "BUSINESS-DEVELOPMENT": {
        "high":   ["lead generation", "sales strategy", "client acquisition", "negotiation",
                   "market research", "revenue growth"],
        "medium": ["partnership development", "account management", "competitor analysis",
                   "proposal writing", "networking", "pipeline management"],
        "tools":  ["salesforce", "hubspot", "linkedin sales navigator", "crm", "ms powerpoint"],
    },

    "CHEF": {
        "high":   ["food preparation", "menu planning", "kitchen management", "food safety",
                   "culinary techniques", "inventory management"],
        "medium": ["baking", "pastry", "cost control", "staff training",
                   "portion control", "recipe development"],
        "tools":  ["haccp", "point of sale systems", "kitchen display systems"],
    },

    "CONSTRUCTION": {
        "high":   ["project management", "site supervision", "structural engineering",
                   "blueprint reading", "cost estimation", "safety compliance"],
        "medium": ["civil engineering", "quantity surveying", "contract management",
                   "quality control", "subcontractor management", "scheduling"],
        "tools":  ["autocad", "revit", "ms project", "primavera", "staad pro"],
    },

    "CONSULTANT": {
        "high":   ["business analysis", "strategic planning", "stakeholder management",
                   "process improvement", "change management", "problem solving"],
        "medium": ["project management", "data analysis", "presentation skills",
                   "workshop facilitation", "report writing", "client management"],
        "tools":  ["ms powerpoint", "tableau", "excel", "jira", "confluence"],
    },

    "DESIGNER": {
        "high":   ["ui design", "ux design", "wireframing", "prototyping",
                   "visual design", "typography", "brand identity"],
        "medium": ["user research", "design systems", "responsive design",
                   "motion graphics", "illustration", "accessibility"],
        "tools":  ["figma", "adobe xd", "photoshop", "illustrator", "sketch", "invision"],
    },

    "DIGITAL-MEDIA": {
        "high":   ["social media marketing", "content creation", "seo", "digital advertising",
                   "analytics", "campaign management"],
        "medium": ["email marketing", "influencer marketing", "video production",
                   "copywriting", "audience targeting", "a/b testing"],
        "tools":  ["google analytics", "meta ads", "hootsuite", "canva", "semrush", "mailchimp"],
    },

    "ENGINEERING": {
        "high":   ["mechanical design", "cad design", "product development", "manufacturing",
                   "quality assurance", "process engineering"],
        "medium": ["project management", "r&d", "simulation", "testing & validation",
                   "supplier management", "lean manufacturing"],
        "tools":  ["solidworks", "autocad", "ansys", "matlab", "catia", "ms project"],
    },

    "FINANCE": {
        "high":   ["financial modeling", "equity research", "portfolio management",
                   "valuation", "investment analysis", "dcf analysis"],
        "medium": ["derivatives", "fixed income", "mutual funds",
                   "mergers and acquisitions", "risk management", "financial planning"],
        "tools":  ["bloomberg terminal", "excel", "python", "r", "power bi", "vba"],
    },

    "FITNESS": {
        "high":   ["personal training", "fitness assessment", "workout programming",
                   "nutrition planning", "client coaching", "group fitness"],
        "medium": ["weight management", "rehabilitation", "sports performance",
                   "flexibility training", "cardio training", "strength training"],
        "tools":  ["trainerize", "myfitnesspal", "mindbody", "heart rate monitors"],
    },

    "HEALTHCARE": {
        "high":   ["patient care", "clinical assessment", "medical diagnosis",
                   "treatment planning", "medical documentation", "emergency care"],
        "medium": ["pharmacology", "infection control", "patient education",
                   "ward management", "triage", "medical ethics"],
        "tools":  ["epic", "meditech", "ehr systems", "medical billing software", "lims"],
    },

    "HR": {
        "high":   ["recruitment", "talent acquisition", "onboarding", "performance management",
                   "employee relations", "hr policy"],
        "medium": ["training and development", "compensation and benefits", "succession planning",
                   "hr analytics", "organizational development", "exit management"],
        "tools":  ["workday", "sap hr", "darwinbox", "greythr", "bamboohr", "zoho people"],
    },

    "INFORMATION-TECHNOLOGY": {
        "high":   ["python", "java", "sql", "software development", "data structures",
                   "algorithms", "rest api", "system design"],
        "medium": ["machine learning", "data analysis", "cloud computing", "cybersecurity",
                   "web development", "database management", "agile", "devops"],
        "tools":  ["aws", "docker", "kubernetes", "git", "linux", "jenkins", "terraform"],
    },

    "PUBLIC-RELATIONS": {
        "high":   ["media relations", "press release writing", "crisis communication",
                   "brand reputation", "stakeholder communication", "event management"],
        "medium": ["social media management", "corporate communications", "content strategy",
                   "speech writing", "media monitoring", "campaign planning"],
        "tools":  ["cision", "meltwater", "prowly", "canva", "google analytics"],
    },

    "SALES": {
        "high":   ["sales strategy", "lead generation", "cold calling", "client relationship",
                   "closing deals", "revenue targets"],
        "medium": ["territory management", "upselling", "cross selling", "negotiation",
                   "sales forecasting", "product demonstration"],
        "tools":  ["salesforce", "hubspot", "zoho crm", "ms excel", "linkedin"],
    },

    "TEACHER": {
        "high":   ["lesson planning", "curriculum development", "classroom management",
                   "student assessment", "instructional design", "subject matter expertise"],
        "medium": ["differentiated instruction", "student counseling", "parent communication",
                   "e-learning", "mentoring", "report writing"],
        "tools":  ["google classroom", "zoom", "ms teams", "moodle", "canva", "smart board"],
    },
}

# ─────────────────────────────────────────────────────────────
# Flatten for backward compatibility if needed
# app.py uses the rich dict directly; this flat version is a bonus
# ─────────────────────────────────────────────────────────────
SKILL_DB_FLAT = {
    role: data["high"] + data["medium"] + data["tools"]
    for role, data in SKILL_TAXONOMY.items()
}

pickle.dump(SKILL_TAXONOMY, open("model/skill_db.pkl", "wb"))
pickle.dump(SKILL_DB_FLAT,  open("model/skill_db_flat.pkl", "wb"))

print("✅ Skill database built and saved!")
print(f"   Roles covered : {len(SKILL_TAXONOMY)}")
for role, data in SKILL_TAXONOMY.items():
    total = len(data["high"]) + len(data["medium"]) + len(data["tools"])
    print(f"   {role:<25} → {total} skills")
