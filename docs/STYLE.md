# Writing style for user-facing text

Applies to every string the browser renders: view and component copy in `web/js/`, policy
cards and capability summaries served from Python, and error messages.

It does not apply to code comments, which explain decisions to the next maintainer and have
a different job.

## Rules

**Avoid em dashes.** Rewrite the sentence with a comma, semicolon, parentheses, or full stop.

**Lead with useful information.** Start with the fact, action, or definition the reader needs
for that section. Bold a term when it helps readers find its definition. Avoid rhetorical
openings and claims about why the reader should care.

**Declarative sentences.** Subject, verb, object. One idea per sentence, one topic per
paragraph. Three or four sentences per paragraph is the ceiling.

**Use direct instructions where useful.** Address the reader when it makes an instruction
clearer. Use a neutral description for reference material and scientific claims.

**Cut filler and editorialising.** Remove phrases such as "crucially", "worth noting", and
"the point is". State the useful fact directly.

**Keep one idea per sentence.** State a conclusion and its supporting evidence plainly.

**Attribute every number.** Say where it came from: measured here, retrieved from a named
database on a named date, or a literature figure. A quantity with no source does not belong
in user-facing text.

**Prefer structure to repeated prose.** Use a table, list, or labelled field when it makes
information easier to scan. Keep paragraphs when the explanation needs connected reasoning.

**Hedge only where there is real uncertainty.** "May", "might" and "could" are for things
that are actually uncertain, not for softening a claim.

## Examples

| Instead of | Write |
|---|---|
| "Potency is the whole story — and the only thing that matters." | "Potency is one factor in dose selection." |
| "That is the exchange rate between an energy column and biology — and it also sets how much of it you are allowed to believe." | "One factor of ten in affinity corresponds to 1.364 kcal/mol at 298 K." |
| "Worth knowing, because it costs an afternoon otherwise: scroll events are tied to the rendering lifecycle." | "Scroll events are delivered as part of the rendering lifecycle." |
| "The classic output is a molecule with an excellent score, an impossible ring fusion and no route." | "Such molecules often combine a high score with an unavailable synthetic route." |
