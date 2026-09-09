"""Entity pools used to fill slots in scaffold and payload templates.

Diversity in the corpus comes from two places: which paragraph templates a
document is built from, and which entities fill their slots. The second matters
more than it looks. Without it, every runbook in the corpus would name the same
service, and the classifier could learn that name as a feature of the scaffold
rather than learning anything about injection.

All values are English, and deliberately mundane: the point is realistic filler,
not interesting content.
"""

SERVICES = (
    "ingest-worker",
    "billing-api",
    "search-indexer",
    "notify-relay",
    "auth-gateway",
    "report-builder",
    "media-transcoder",
    "ledger-sync",
    "quota-manager",
    "session-store",
)

TEAMS = (
    "platform",
    "data services",
    "customer operations",
    "site reliability",
    "internal tooling",
    "security engineering",
    "finance systems",
    "integrations",
)

PEOPLE = (
    "A. Bello",
    "R. Danjuma",
    "S. Okonkwo",
    "M. Yusuf",
    "T. Adeyemi",
    "K. Lawal",
    "N. Eze",
    "H. Suleiman",
    "J. Mwangi",
    "F. Balogun",
)

ENVIRONMENTS = ("staging", "production", "pre-production", "sandbox", "canary")

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "June",
    "July",
    "September",
    "October",
    "November",
)

DAYS = ("3rd", "7th", "11th", "14th", "18th", "22nd", "26th", "29th")

SMALL_NUMBERS = ("two", "three", "four", "five", "six", "eight", "ten", "twelve")

DIGITS = ("2", "3", "4", "5", "6", "8", "12", "15", "20", "30")

VERSIONS = ("1.4", "2.0", "2.3", "3.1", "3.7", "4.2", "5.0", "6.1")

TICKETS = (
    "OPS-1182",
    "OPS-2043",
    "SUP-3391",
    "SUP-4417",
    "INT-0928",
    "INT-1355",
    "SEC-7714",
    "SEC-8206",
)

REGIONS = ("eu-west", "us-east", "af-south", "ap-south", "us-central")

DOCUMENT_TOPICS = (
    "retention",
    "access review",
    "capacity planning",
    "vendor onboarding",
    "cost allocation",
    "incident response",
    "release readiness",
    "data classification",
)

# Title-cased variants. Document titles are capitalised, and dropping a
# lower-case slot value into one produces "Revised cost allocation Policy",
# which reads as generated text rather than as a real document. Templates use
# these in titles and the plain pools in body prose.
TOPICS_TITLE_CASE = tuple(topic.title() for topic in DOCUMENT_TOPICS)
TEAMS_TITLE_CASE = tuple(team.title() for team in TEAMS)

#: Slot name to pool. Templates reference these with ``str.format`` syntax,
#: for example ``"The {service} deployment in {environment} ..."``. A
#: placeholder with no entry here is left untouched by the generator.
SLOT_POOLS: dict[str, tuple[str, ...]] = {
    "service": SERVICES,
    "team": TEAMS,
    "team_tc": TEAMS_TITLE_CASE,
    "person": PEOPLE,
    "environment": ENVIRONMENTS,
    "month": MONTHS,
    "day": DAYS,
    "count": SMALL_NUMBERS,
    "digits": DIGITS,
    "version": VERSIONS,
    "ticket": TICKETS,
    "region": REGIONS,
    "topic": DOCUMENT_TOPICS,
    "topic_tc": TOPICS_TITLE_CASE,
}
