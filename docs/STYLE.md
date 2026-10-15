# Writing style for user-facing text

Applies to every string the browser renders: view and component copy in `web/js/`, policy
cards and capability summaries served from Python, and error messages.

It does not apply to code comments, which explain decisions to the next maintainer and have
a different job.

## Rules

**No em dashes.** Use a comma, a semicolon, parentheses, or a full stop. An em dash almost
always marks a clause that was appended as an afterthought; splitting the sentence is both
clearer and shorter.

**Lead with a definition.** The first sentence of a section says what the subject is, in the
form `X is ...`. Bold the term on its first appearance. Do not open with a question, a
scene, or a claim about why the reader should care.

**Declarative sentences.** Subject, verb, object. One idea per sentence, one topic per
paragraph. Three or four sentences per paragraph is the ceiling.

**No second person.** Write "a researcher sets the cutoff" or "the cutoff is set", not "you
set the cutoff". The exception is a numbered procedure, where the imperative is correct:
"Enable retrieval in step 3."

**No editorialising.** Cut "crucially", "importantly", "worth noting", "the point is",
"which is exactly why", "the part usually left out". If a fact matters, state the fact. If
it needs emphasis, give it its own sentence.

**No rhetorical build-up.** State the conclusion, then the evidence. Do not withhold a
result for effect.

**Attribute every number.** Say where it came from: measured here, retrieved from a named
database on a named date, or a literature figure. A quantity with no source does not belong
in user-facing text.

**Prefer structure to prose.** A table, a list, or a labelled field beats a paragraph that
describes the same thing. Reserve paragraphs for reasoning that genuinely runs in sequence.

**Hedge only where there is real uncertainty.** "May", "might" and "could" are for things
that are actually uncertain, not for softening a claim.

## Examples

| Instead of | Write |
|---|---|
| "Potency is not a virtue in itself — it is what buys you a dose small enough for the selectivity you have." | "Potency is a means to an adequate dose rather than an end in itself." |
| "That is the exchange rate between an energy column and biology — and it also sets how much of it you are allowed to believe." | "One factor of ten in affinity corresponds to 1.364 kcal/mol at 298 K." |
| "Worth knowing, because it costs an afternoon otherwise: scroll events are tied to the rendering lifecycle." | "Scroll events are delivered as part of the rendering lifecycle." |
| "The classic output is a molecule with an excellent score, an impossible ring fusion and no route." | "Such molecules often combine a high score with an unavailable synthetic route." |
