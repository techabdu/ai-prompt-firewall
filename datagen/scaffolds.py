"""Document scaffolds: the ordinary business and technical documents that carry
the corpus.

Every document in the corpus, benign or malicious, is built on one of these.
That is deliberate and it is the single most important structural property of
the dataset. If malicious documents were all runbooks and benign ones all
meeting minutes, a classifier could reach near-perfect scores by recognising
the template and learning nothing whatsoever about injection. Sharing the
scaffold pool evenly across both labels removes that shortcut, and
``tests/test_dataset.py`` asserts the parity holds.

Ten scaffolds build the main corpus. Four more are marked ``held_out`` and
appear only in the challenge set, which asks a harder question: does the
detector generalise to document types it has never seen, or has it learned
these ten?

Paragraph text is templated with slots drawn from ``datagen.slots``. Prose is
intentionally unremarkable -- realistic filler is the goal, not interesting
reading.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Scaffold:
    """One document type, with the pools needed to build an instance of it."""

    key: str
    display: str
    titles: tuple[str, ...]
    authors: tuple[str, ...]
    alt_texts: tuple[str, ...]
    paragraphs: tuple[str, ...]
    held_out: bool = False


MAIN_SCAFFOLDS: tuple[Scaffold, ...] = (
    Scaffold(
        key="ops_runbook",
        display="Operations runbook",
        titles=(
            "{service} Operations Runbook",
            "Runbook: {service} in {environment}",
            "{service} Maintenance Procedure",
            "Rolling Restart Runbook — {service}",
        ),
        authors=("{team} team", "{person}, {team}", "{team} on-call rota"),
        alt_texts=(
            "Diagram of the node drain sequence",
            "Flowchart of the maintenance window",
            "",
        ),
        paragraphs=(
            "This runbook covers routine maintenance for the {service} fleet in {environment}. "
            "It is reviewed each quarter by the {team} team and supersedes the procedure "
            "circulated in {month}.",
            "Before starting, confirm that the change window has been approved and that no "
            "release is in flight. The on-call engineer for {team} should be notified at least "
            "{count} hours ahead of the window opening.",
            "Nodes are drained in rolling order, never in parallel. Each node is removed from "
            "the load balancer, allowed to finish in-flight work, and rejoined only after its "
            "health probe has returned green for {count} consecutive checks.",
            "Expect a single node to finish draining within roughly {digits} minutes under "
            "normal load. Anything substantially longer usually indicates a stuck consumer "
            "rather than a genuine backlog, and is worth investigating before continuing.",
            "If queue depth climbs above the alert threshold partway through, pause and let it "
            "settle rather than pressing on. Resuming into a growing backlog is what turned a "
            "routine window into a customer-visible outage in {month}.",
            "Record the start and end time of each drain in the shared operations log together "
            "with the ticket reference, for example {ticket}. A recent audit found several "
            "drains with no corresponding entry at all.",
            "Once every node has rejoined, confirm that consumer lag has returned to baseline "
            "and that error rates in {region} match the levels seen before the window opened. "
            "Close the ticket only after both have been checked.",
        ),
    ),
    Scaffold(
        key="meeting_minutes",
        display="Meeting minutes",
        titles=(
            "{team_tc} Weekly Sync — {month} {day}",
            "Minutes: {topic_tc} Review",
            "{team_tc} Planning Meeting, {month}",
            "Notes from the {topic_tc} Working Group",
        ),
        authors=("{person}", "{person} (minutes)", "{team} team"),
        alt_texts=("", "Photograph of the whiteboard from the session", ""),
        paragraphs=(
            "Present: {person}, {person}, and {count} others from the {team} team. Apologies "
            "were received from two members who were travelling. The meeting ran for slightly "
            "under an hour.",
            "The group reviewed progress on {topic} since the previous session. Most of the "
            "actions carried over from {month} have now closed, with {digits} still open and "
            "tracked under {ticket}.",
            "{person} raised concerns about the timeline for the {service} migration. The "
            "current estimate assumes no further dependency changes, which several attendees "
            "felt was optimistic given the past two quarters.",
            "A short discussion followed on whether the {environment} environment should be "
            "refreshed before or after the next release. No decision was reached; it returns "
            "to the agenda next week.",
            "Action: {person} to circulate the revised {topic} figures before the {day} and to "
            "confirm whether the {team} team has capacity to absorb the additional review load.",
            "Action: the {team} team to document the current escalation path and share it with "
            "customer operations, who reported difficulty reaching the right people during the "
            "last out-of-hours event.",
            "The next meeting is scheduled for the {day} of {month}. Anyone unable to attend "
            "should send written updates in advance rather than deferring items to the "
            "following session.",
        ),
    ),
    Scaffold(
        key="policy_memo",
        display="Internal policy memo",
        titles=(
            "Policy Update: {topic_tc}",
            "Internal Memo — {topic_tc} Standards",
            "Revised {topic_tc} Policy, effective {month}",
            "{topic_tc}: Interim Guidance",
        ),
        authors=("{team} team", "{person}, Head of {team}", "Policy Office"),
        alt_texts=("", "", "Table summarising the policy change"),
        paragraphs=(
            "This memo sets out revised guidance on {topic}, effective from the {day} of "
            "{month}. It replaces the interim guidance issued last year and applies to all "
            "staff in the {team} function.",
            "The change follows a review that found inconsistent practice across teams. In "
            "several cases the same activity was being handled {count} different ways "
            "depending on who was asked, which made assurance work considerably harder.",
            "Under the revised policy, requests must be recorded in the central register within "
            "{digits} working days. Retrospective entries will be accepted only where a "
            "documented exception applies.",
            "Line managers are responsible for confirming that their teams have read this memo. "
            "Confirmation should be logged against {ticket} rather than sent by email, so that "
            "there is a single auditable record.",
            "Existing arrangements entered into before {month} continue under the previous terms "
            "until their scheduled review date. There is no requirement to reopen them early.",
            "Questions about interpretation should go to the {team} team in the first instance. "
            "Where a case is genuinely ambiguous, it will be logged and used to inform the next "
            "revision rather than settled informally.",
            "This guidance will be reviewed again within twelve months, or sooner if the "
            "regulatory position changes. A short summary of any amendments will be circulated "
            "to all affected staff in {region}.",
        ),
    ),
    Scaffold(
        key="support_ticket",
        display="Support ticket thread",
        titles=(
            "{ticket}: {service} errors reported by customer",
            "Ticket {ticket} — intermittent failures in {region}",
            "{ticket}: escalation from customer operations",
            "Support thread: {service} timeouts",
        ),
        authors=("Customer Operations", "{person}, support", "{team} triage"),
        alt_texts=("", "Screenshot supplied by the reporting customer", ""),
        paragraphs=(
            "Customer reports intermittent failures when calling the {service} endpoint from "
            "{region}. The failures began around the {day} and affect roughly one request in "
            "{digits}, with no obvious pattern by time of day.",
            "First response from {person}: asked the customer to supply request identifiers for "
            "{count} recent failures and to confirm which client library version they are "
            "running. Response received the same afternoon.",
            "Triage note: the supplied identifiers all resolve to the same upstream node. That "
            "narrows this considerably and suggests a single unhealthy instance rather than a "
            "systemic problem with the service.",
            "Update from the {team} team: the node in question was replaced during the {month} "
            "window and has not shown the same behaviour since. Monitoring has been left in "
            "place for a further week as a precaution.",
            "Customer confirms the error rate has dropped to zero over the last {count} days. "
            "They have asked whether anything is needed on their side; the answer is no, but "
            "upgrading the client remains advisable.",
            "Internal note: this is the second ticket this quarter tracing back to a single "
            "unhealthy instance surviving a health check. Worth raising with {team} as a "
            "possible gap in the probe configuration rather than treating each case separately.",
            "Ticket set to resolved pending customer confirmation. If no response is received "
            "within {digits} working days it will close automatically and can be reopened on "
            "request.",
        ),
    ),
    Scaffold(
        key="wiki_page",
        display="Internal wiki page",
        titles=(
            "{service} — Internal Reference",
            "{topic_tc}: How It Works",
            "Wiki: {service} architecture notes",
            "{team_tc} Team Handbook — {topic_tc}",
        ),
        authors=("{team} team", "{person}", "maintained by {team}"),
        alt_texts=("Architecture diagram", "", "Component overview diagram"),
        paragraphs=(
            "This page describes how {service} fits into the wider platform. It is maintained "
            "by the {team} team and was last reviewed in {month}. Corrections are welcome; "
            "please raise them rather than editing structural sections directly.",
            "The service consumes from {count} upstream queues and writes to the primary store "
            "in {region}. A secondary replica exists for read traffic but is not currently used "
            "for failover.",
            "Configuration is held in the shared repository and applied at deploy time. There is "
            "no runtime configuration reload, so a change requires a rolling restart to take "
            "effect across the fleet.",
            "Capacity has been sized against the peak observed in {month}, with roughly {digits} "
            "per cent headroom. That margin has been adequate so far, though it narrows during "
            "end-of-quarter processing.",
            "Known limitations are tracked under {ticket}. The most significant is that retries "
            "are not idempotent for one class of request, which means a duplicate can occasionally "
            "reach the downstream consumer.",
            "For historical context: this service was split out of the monolith in version "
            "{version}, largely to isolate its scaling profile. The original design notes are "
            "archived and linked at the foot of this page.",
            "If you are new to this area, start with the {environment} environment rather than "
            "reading the code. Following a single request end to end is considerably faster than "
            "any written explanation.",
        ),
    ),
    Scaffold(
        key="incident_postmortem",
        display="Incident post-mortem",
        titles=(
            "Post-mortem: {service} outage, {month} {day}",
            "Incident Review — {ticket}",
            "{service} Degradation: What Happened",
            "Post-incident Report, {month}",
        ),
        authors=("{team} team", "{person}, incident lead", "Site Reliability"),
        alt_texts=("Timeline graph of the incident", "", "Error rate during the window"),
        paragraphs=(
            "On the {day} of {month}, {service} returned elevated error rates for approximately "
            "{digits} minutes. Customers in {region} were affected. This review is blameless and "
            "is intended to identify contributing factors, not individuals.",
            "The trigger was a configuration change applied to {environment} that was intended "
            "to be limited in scope. It was not. The change propagated further than the author "
            "expected, which the review team considers a tooling failure rather than a human one.",
            "Detection took longer than it should have. The first alert fired {count} minutes "
            "after the error rate began climbing, because the threshold was tuned for a slower "
            "class of failure than the one that occurred.",
            "Mitigation was straightforward once the cause was identified: the change was rolled "
            "back and the fleet restarted. Recovery was complete within a further {count} "
            "minutes, with no data loss.",
            "Contributing factor: the change was applied outside the normal window, so fewer "
            "people were watching than usual. The reviewing engineer had approved a similar "
            "change previously and did not re-read the scope.",
            "Action: tighten the alert threshold and add a scope check to the deployment tool, "
            "tracked under {ticket}. Both are owned by the {team} team and due before the end of "
            "{month}.",
            "Action: add this scenario to the {environment} game-day exercises. The team handled "
            "it well, but the response depended on one person recognising the pattern, which is "
            "not something to rely on.",
        ),
    ),
    Scaffold(
        key="changelog",
        display="Product changelog",
        titles=(
            "{service} {version} — Release Notes",
            "Changelog: {service}, version {version}",
            "Release {version} Notes",
            "What's New in {service} {version}",
        ),
        authors=("{team} team", "Release Management", "{person}"),
        alt_texts=("", "", "Screenshot of the updated interface"),
        paragraphs=(
            "Version {version} of {service} is now available in {environment} and will reach "
            "general availability on the {day} of {month}. This is a minor release with no "
            "breaking changes to the public interface.",
            "Added: support for filtering results by region, which had been requested "
            "repeatedly and is tracked under {ticket}. Filtering applies to both the list and "
            "export paths.",
            "Changed: the default request timeout has been raised, following analysis showing "
            "that a small proportion of legitimate requests in {region} were being cut short "
            "under load.",
            "Fixed: a defect where retried requests could occasionally produce a duplicate entry "
            "downstream. The fix has been running in {environment} for {count} weeks without "
            "recurrence.",
            "Fixed: several inconsistencies in error messages, which previously made it hard to "
            "distinguish a client-side validation failure from an upstream problem.",
            "Deprecated: the legacy export format remains available but will be withdrawn in "
            "version {version}. Consumers should migrate before then; the {team} team can "
            "advise on the transition.",
            "Upgrade notes: no configuration changes are required. Deployments should follow the "
            "standard rolling procedure, allowing roughly {digits} minutes for the full fleet.",
        ),
    ),
    Scaffold(
        key="onboarding_guide",
        display="Onboarding guide",
        titles=(
            "Onboarding: {team_tc} Team",
            "New Starter Guide — {team_tc}",
            "Your First Weeks in {team_tc}",
            "{team_tc} Onboarding Handbook",
        ),
        authors=("{team} team", "People Operations", "{person}"),
        alt_texts=("", "Org chart for the team", ""),
        paragraphs=(
            "Welcome to the {team} team. This guide covers your first few weeks and is written "
            "to be read once and then referred back to, so do not try to absorb all of it on "
            "your first day.",
            "Your manager will arrange access to the systems you need. If something is still "
            "missing after {count} days, raise it rather than working around it; missing access "
            "is the single most common cause of a slow start here.",
            "In the first week, focus on {topic}. Most of what the team does touches it in some "
            "form, and understanding it early makes the rest considerably easier to follow.",
            "You will be paired with a buddy from the team for your first month. They are there "
            "for the questions that feel too small to ask in a meeting, which are usually the "
            "ones worth asking.",
            "The team runs a weekly sync on the {day}. Attendance is expected but participation "
            "is not, at least not initially. Listening for a few weeks is a perfectly reasonable "
            "way to start.",
            "Documentation lives in the internal wiki and is of variable quality. If you find "
            "something wrong or out of date, correcting it is genuinely welcome and is one of "
            "the more useful things a new starter can do.",
            "By the end of your first {count} weeks you should have made a small change end to "
            "end, from ticket to deployment. Your manager will help pick something suitably "
            "contained, such as {ticket}.",
        ),
    ),
    Scaffold(
        key="faq_page",
        display="FAQ page",
        titles=(
            "{topic_tc}: Frequently Asked Questions",
            "FAQ — {service}",
            "Common Questions About {topic_tc}",
            "{service} FAQ, updated {month}",
        ),
        authors=("{team} team", "Customer Operations", "{person}"),
        alt_texts=("", "", "Illustration of the request flow"),
        paragraphs=(
            "This page answers the questions the {team} team is asked most often about {topic}. "
            "It was last updated in {month}. If your question is not answered here, raise a "
            "ticket rather than asking in a channel.",
            "How long does a request usually take? Most complete within {digits} seconds. "
            "Requests that take substantially longer are almost always waiting on an upstream "
            "dependency rather than doing work themselves.",
            "Can I use this in {environment}? Yes, though the data there is refreshed "
            "periodically and should not be relied on for anything that needs to persist beyond "
            "a few weeks.",
            "Why did my request fail with no error detail? This usually means the request was "
            "rejected before reaching the service. Check the client library version first; "
            "older versions discard the detail.",
            "Who owns this service? The {team} team. For anything urgent outside working hours, "
            "use the on-call rota rather than contacting individuals, who may not be the person "
            "carrying the pager.",
            "Is there a limit on request size? Yes. Requests above the documented limit are "
            "rejected outright. The limit was raised in version {version} and is unlikely to "
            "change again soon.",
            "How do I report a problem? Open a ticket with the request identifier and the "
            "approximate time. Those two details resolve most reports without any further "
            "correspondence.",
        ),
    ),
    Scaffold(
        key="vendor_questionnaire",
        display="Vendor security questionnaire response",
        titles=(
            "Security Questionnaire Response — {topic_tc}",
            "Vendor Assessment Reply, {month}",
            "{topic_tc}: Supplier Assurance Response",
            "Completed Security Questionnaire",
        ),
        authors=("{team} team", "{person}, Security Engineering", "Assurance Office"),
        alt_texts=("", "Diagram of the data flow between parties", ""),
        paragraphs=(
            "This response covers the questionnaire received on the {day} of {month} concerning "
            "{topic}. Answers reflect the position as at the date of writing and will be "
            "restated at the next annual review.",
            "Data residency: customer data is held in {region} and is not replicated outside "
            "that region. A secondary copy exists within the same region for resilience "
            "purposes only.",
            "Access control: access is granted on a least-privilege basis and reviewed every "
            "{count} months. The most recent review completed in {month} and closed all "
            "outstanding exceptions.",
            "Subprocessors: {digits} subprocessors are engaged, each covered by a written "
            "agreement. A current list is maintained and can be supplied on request under the "
            "existing confidentiality terms.",
            "Incident notification: material incidents affecting customer data are notified "
            "within the contractually agreed period. The notification path is tested annually "
            "as part of the {environment} exercise programme.",
            "Certification: the relevant certification is held and was renewed in {month}. A "
            "copy of the current certificate and the accompanying statement of applicability "
            "can be provided.",
            "Outstanding items: {count} questions in section four require input from the {team} "
            "team and will follow separately. They are tracked under {ticket} and are not "
            "expected to change the overall position.",
        ),
    ),
)


HELD_OUT_SCAFFOLDS: tuple[Scaffold, ...] = (
    Scaffold(
        key="research_abstract",
        display="Research paper abstract and summary",
        held_out=True,
        titles=(
            "Abstract: An Empirical Study of {topic_tc}",
            "Summary — {topic_tc} in Distributed Systems",
            "Research Note on {topic_tc}",
        ),
        authors=("{person} et al.", "{person} and {person}", "Research Group"),
        alt_texts=("", "Figure 1: results by condition", ""),
        paragraphs=(
            "This paper examines {topic} across {digits} organisations over an eighteen-month "
            "period. Prior work has largely relied on self-reported survey data; we instead "
            "draw on operational records, which we argue gives a more reliable picture.",
            "Our method combines a quantitative analysis of incident records with {count} "
            "semi-structured interviews. Participants were drawn from teams of varying size to "
            "avoid over-representing large organisations.",
            "We find a consistent gap between documented process and observed practice. In the "
            "majority of cases the documented process was followed only when an incident was "
            "already under way, rather than as routine.",
            "The results suggest that process documentation alone is a poor predictor of "
            "behaviour. Teams that rehearsed their procedures, even informally, diverged far "
            "less than those that had merely written them down.",
            "We conclude with {count} recommendations for practitioners and note the principal "
            "limitation of this work: our sample is drawn from a single sector, and "
            "generalisation beyond it should be treated with caution.",
        ),
    ),
    Scaffold(
        key="job_description",
        display="Job description",
        held_out=True,
        titles=(
            "Job Description — Engineer, {team_tc}",
            "Vacancy: {team_tc} Specialist",
            "Role Profile: {topic_tc} Lead",
        ),
        authors=("People Operations", "{team} team", "Recruitment"),
        alt_texts=("", "", "Photograph of the office"),
        paragraphs=(
            "We are recruiting an engineer to join the {team} team, based in {region} with "
            "flexible working arrangements. The role reports to the team lead and works closely "
            "with customer operations.",
            "The successful candidate will take ownership of {topic}, working across "
            "{count} services including {service}. Day-to-day work is a mixture of "
            "improvement work and responding to what the platform actually needs that week.",
            "We are looking for someone comfortable with ambiguity and willing to write things "
            "down. Formal qualifications matter less to us than evidence of having improved "
            "something and being able to explain why it needed improving.",
            "The team operates an on-call rota shared between {count} engineers, with "
            "compensation as set out in the standard terms. New joiners are not added to the "
            "rota until they are ready.",
            "Applications close on the {day} of {month}. We aim to respond to every applicant, "
            "and will provide feedback to anyone who reaches interview stage regardless of the "
            "outcome.",
        ),
    ),
    Scaffold(
        key="expense_policy",
        display="Travel and expense policy",
        held_out=True,
        titles=(
            "Travel and Expense Policy, {month}",
            "Expenses Guidance for {team_tc}",
            "Business Travel Policy — Revised",
        ),
        authors=("Finance Systems", "{person}, Finance", "Policy Office"),
        alt_texts=("", "Table of per-diem rates", ""),
        paragraphs=(
            "This policy applies to all business travel booked from the {day} of {month}. It "
            "replaces the previous version and should be read before booking rather than after "
            "returning.",
            "Travel must be booked through the approved provider wherever possible. Direct "
            "bookings are reimbursed only where the approved provider could not offer a "
            "comparable option, and require a short written explanation.",
            "Claims must be submitted within {digits} working days of return, with receipts "
            "attached. Claims submitted later than that require sign-off from a second approver "
            "in the {team} function.",
            "Subsistence is reimbursed at the published rate for the destination region rather "
            "than on an actual-cost basis. The rates were last revised in {month} and are "
            "reviewed annually.",
            "Questions should be directed to Finance Systems. Where a claim is genuinely "
            "borderline, ask before incurring the cost; retrospective approval is considerably "
            "harder to obtain and occasionally refused.",
        ),
    ),
    Scaffold(
        key="api_integration_guide",
        display="API integration guide",
        held_out=True,
        titles=(
            "Integrating with the {service} API",
            "{service} API — Integration Guide, version {version}",
            "Developer Guide: {service}",
        ),
        authors=("{team} team", "Developer Relations", "{person}"),
        alt_texts=("Sequence diagram of the authentication flow", "", ""),
        paragraphs=(
            "This guide walks through integrating with the {service} API, version {version}. It "
            "assumes familiarity with HTTP APIs but not with our platform specifically, and "
            "works through a complete example.",
            "Authentication uses a token issued per environment. Tokens issued for "
            "{environment} will not work against other environments, which is the most common "
            "cause of the confusing authorisation errors people report.",
            "Requests are rate limited per client. The limit is generous for interactive use "
            "but will be reached by an unthrottled batch job, so batch consumers should apply "
            "a modest delay between calls.",
            "Errors are returned with a machine-readable code and a human-readable message. "
            "Client code should branch on the code; the message text is not stable across "
            "versions and should never be parsed.",
            "For support, open a ticket including the request identifier from the response "
            "headers. That identifier lets the {team} team find the exact request, which is "
            "considerably faster than a description of what went wrong.",
        ),
    ),
)


ALL_SCAFFOLDS: tuple[Scaffold, ...] = MAIN_SCAFFOLDS + HELD_OUT_SCAFFOLDS

SCAFFOLDS_BY_KEY: dict[str, Scaffold] = {s.key: s for s in ALL_SCAFFOLDS}
