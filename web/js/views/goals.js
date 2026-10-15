/* Goals in drug discovery, as a reference article.
 *
 * Laid out as a wiki page rather than as part of the application: a contents rail, numbered
 * sections, a measured reading column, everything expanded. The previous version hid each
 * goal type behind an accordion, which is the right shape for a control panel and the wrong
 * one for a document: collapsed cards cannot be skimmed, cannot be found by in-page search,
 * and cannot be linked to.
 *
 * Prose follows docs/STYLE.md -- definition first, declarative sentences, no em dashes, no
 * second person outside a numbered procedure. The earlier draft was written as an essay and
 * read as one, with the conclusion of each paragraph withheld until its last clause.
 *
 * Every quoted number is either a measurement from a public bank, computed here, or flagged
 * as a literature figure. Nothing comes from a run of this framework: a page about how to
 * judge results should not calibrate the reader against output from the tool whose results
 * they are learning to doubt. The ligand-efficiency figures come from measured ChEMBL
 * affinities because LE is defined on experimental free energy; computing it from a docking
 * score reproduces the size dependence the metric exists to remove.
 */

import { useEffect, useState } from 'preact/hooks';
import { html } from '@/h.js';
import { SelectionFigure } from '@/components/SelectionFigure.js';
import { loadGlossary } from '@/components/MetricTerm.js';

/* Where a goal clicked here is parked while the setup page loads. Session, not local: a goal
 * clicked an hour ago in another tab should not reappear in this one. */
export const GOAL_HANDOFF = 'drugorch-goal-draft';

/* ---------------------------------------------------------------------------------------
 * Measured reference data.
 *
 * Retrieved from ChEMBL with this project's own client on 2026-10-14; the query is in
 * PROVENANCE so it can be rerun. Baked in rather than fetched on page load because a
 * reference article should not depend on a third-party endpoint being reachable.
 * ------------------------------------------------------------------------------------ */

const LE_ROWS = [
  {
    id: 'CHEMBL1789941',
    note: 'ruxolitinib',
    assay: 'Kd',
    pchembl: 10.44,
    hac: 23,
    le: 0.619,
    logp: 3.47,
  },
  { id: 'CHEMBL1938649', note: '', assay: 'IC50', pchembl: 7.24, hac: 20, le: 0.494, logp: 2.91 },
  { id: 'CHEMBL231733', note: '', assay: 'IC50', pchembl: 7.28, hac: 39, le: 0.255, logp: 3.89 },
];

const LE_STATS = {
  n: 400,
  target: 'CHEMBL2971',
  medianHac: 30,
  hacRange: '17 to 43',
  medianLe: 0.372,
  medianP: 8.0,
  corrSizePotency: 0.17,
  corrSizeLe: -0.859,
};

const PROVENANCE =
  'ChEMBL target CHEMBL2971 (JAK2), activities of type IC50, Ki, Kd and EC50 with '
  + 'pChEMBL at or above 5, best 400, retrieved 2026-10-14. LE = 1.364 × pChEMBL / heavy '
  + 'atoms. Heavy-atom counts and cLogP from RDKit.';

/* ---------------------------------------------------------------------------------------
 * The catalogue. `needs` are subsystem keys, cross-referenced against the live manifest.
 * ------------------------------------------------------------------------------------ */

const GOALS = [
  {
    key: 'potency',
    reach: 'full',
    name: 'Potency',
    tag: 'strength of binding',

    definition: `**Potency** is the concentration at which a compound produces a given effect
                 on its target. It is measured as IC50, Ki, Kd or EC50, and commonly quoted as
                 pChEMBL, the negative base-10 logarithm of the molar value. pChEMBL 6
                 corresponds to 1 µM, 7 to 100 nM and 9 to 1 nM.`,

    significance: `A compound must occupy its target at a concentration the body can sustain.
                   Weak binding requires a larger dose, and a larger dose also occupies the
                   compound's weaker interactions with other proteins. Potency is therefore a
                   means to an adequate dose rather than an end in itself. Binding affinity
                   and binding free energy are equivalent statements of the same quantity; the
                   conversion is given in §3.`,

    methods: `Potency is measured. A biochemical assay gives IC50 or Ki and a cell assay gives
              EC50. Computation selects what to synthesise next and does not establish a
              potency. A series is driven by structure-activity relationships: the analogue is
              prepared, measured, and the direction of the change indicates where the binding
              site is constrained.`,

    here: [
      'Select the target in step 2. The readiness check reports whether measured actives and a usable structure exist before any budget is spent.',
      'Enable retrieval and docking in step 3. Retrieval returns compounds with measured affinity; docking scores any compound, measured or generated.',
      'Read the retrieval parameters in the trace. The pChEMBL cutoff is the decision the policy made, and every later step inherits it.',
      'Compare generated compounds with the retrieved actives from the same run. Both passed through the same scoring function, so the ranking is comparable even though the absolute values are not.',
    ],

    verdict: {
      read: `Relative rank within one run. A generated compound that scores better than every
             retrieved active is a candidate for follow-up. A margin of 0.3 kcal/mol is within
             the error of the scoring function.`,
      fooled: `Ranking by raw score. Docking scores increase with heavy-atom count, so the
               largest compound frequently ranks first without binding more tightly. See §3.`,
    },

    needs: ['retrieval', 'docking'],
    gap: `Docking is a ranking function, not an assay. No part of a run measures affinity, and
          the correlation with measured values is weak enough that a score should not be
          reported as a predicted potency.`,

    examples: [
      {
        text:
          'find low-nanomolar JAK2 inhibitors; treat anything weaker than pChEMBL 7 as a miss',
        note: 'States a failure condition, so the policy can distinguish progress from activity.',
      },
      {
        text:
          'beat the docking score of the best retrieved active by at least 1 kcal/mol in the '
          + 'same pocket and box',
        note: 'Relative, so systematic error cancels. 1 kcal/mol is above the noise floor.',
      },
    ],
  },

  {
    key: 'selectivity',
    reach: 'partial',
    name: 'Selectivity',
    tag: 'activity at other proteins',

    definition: `**Selectivity** is the ratio of a compound's activity at its intended target
                 to its activity at other proteins. It is reported as a ratio of potencies,
                 such as Ki(JAK1)/Ki(JAK2), or as the fraction of a screening panel inhibited.
                 An **anti-target** is a protein the compound must not bind.`,

    significance: `Binding sites are conserved within protein families. The ATP site is similar
                   across the human kinome, so a compound designed for one kinase commonly
                   inhibits others. Ruxolitinib inhibits JAK1 and JAK2. The erythropoietin and
                   thrombopoietin receptors signal through JAK2, so anaemia and
                   thrombocytopenia follow from on-target activity in unintended tissues.
                   JAK1-preferential inhibitors were developed later in part to avoid this.
                   Terfenadine and cisapride were withdrawn over hERG-mediated cardiac
                   arrhythmia, and hERG is now screened early as standard practice.`,

    methods: `Counter-screening. The compound is measured against the intended target and
              against its relatives, and the ratio is reported. Selectivity is designed in by
              identifying residues that differ between the relatives and forming a contact
              only the intended target can make.`,

    here: [
      'Start two runs with the same goal, seed, budget and tools, changing only the target.',
      'Compare the two traces on the Runs page. With identical settings the protein is the only variable.',
      'For a structural anti-target such as hERG, read the property panel instead. A basic amine attached to a lipophilic aromatic system is the common hERG pharmacophore and is visible without docking.',
    ],

    verdict: {
      read: `The difference between two runs with identical settings. A compound scoring −9
             against the intended target and −9 against its relative has shown nothing.`,
      fooled: `Treating a cross-target score difference as a selectivity ratio. Two targets
               require two receptor preparations and two boxes, so systematic errors do not
               cancel. Differences below about 1.5 kcal/mol are unresolved.`,
    },

    needs: ['retrieval', 'docking'],
    gap: `No counter-screen exists. A run has one target, and nothing docks against an
          anti-target or computes a ratio. Stating selectivity in the goal will influence an
          LLM policy's thresholds, but the framework cannot measure whether it was achieved.
          The two-run comparison above is manual.`,

    examples: [
      {
        text:
          'selective for JAK2 over JAK1; a potent pan-JAK compound counts as a failure, not a '
          + 'partial success',
        note: 'States the trade-off as a rule rather than a preference.',
      },
      {
        text:
          'avoid the hERG pharmacophore: no basic amine attached to a lipophilic aromatic '
          + 'system',
        note: 'A structural proxy for an anti-target, visible to the property panel.',
      },
    ],
  },

  {
    key: 'admet',
    reach: 'partial',
    name: 'ADMET and developability',
    tag: 'behaviour in the body',

    definition: `**ADMET** covers absorption, distribution, metabolism, excretion and
                 toxicity: the properties determining whether a compound reaches its target in
                 a patient and is cleared without harm. **Developability** is the broader
                 question of whether a compound can be formulated, manufactured and dosed.`,

    significance: `Analyses of clinical attrition in the 1960s to 1980s attributed roughly 40%
                   of failures to pharmacokinetics and bioavailability. Published analyses of
                   1990s attrition put the figure near 10%, with efficacy and safety dominating
                   instead. The change followed the movement of ADME screening earlier in the
                   pipeline. Both figures are literature values recalled here rather than
                   measured.`,

    methods: `Rule-based triage first. Lipinski's rule of five (MW at or below 500, cLogP at or
              below 5, donors at or below 5, acceptors at or below 10) and Veber's criteria
              (TPSA at or below 140 Å², rotatable bonds at or below 10) estimate oral
              absorption. QED summarises drug-likeness as one value. PAINS flags substructures
              that interfere with assays. Measurement follows: microsomal stability, Caco-2
              permeability, hERG patch clamp, Ames mutagenicity.`,

    here: [
      'Enable properties in step 3. It computes twelve descriptors plus QED, SA score, Lipinski violations and PAINS alerts for every compound the run touches.',
      'Open a properties call in the trace. The explain block reports minimum, median and maximum per descriptor and the number of compounds tripping each flag.',
      'State the numeric envelope in the goal, so an LLM policy has thresholds to apply rather than an adjective to interpret.',
    ],

    verdict: {
      read: `Flag counts rather than averages. "Three of forty compounds trip PAINS"
             identifies specific compounds; a mean QED of 0.6 identifies none.`,
      fooled: `Treating the rules as thresholds for viability. Several approved drugs violate
               Lipinski's criteria, and a PAINS alert indicates an assay to verify rather than
               a compound to discard.`,
    },

    needs: ['properties'],
    gap: `The in-run checks are descriptors and rules, not predictions. Nothing estimates
          half-life, clearance or bioavailability. The project contains a trained ADMET bridge
          covering Ames, blood-brain penetration, logD and aqueous solubility, but it serves
          the benchmark harness and is not a routable subsystem.`,

    examples: [
      {
        text:
          'orally bioavailable: MW under 500, cLogP under 5, TPSA under 140, no PAINS alerts',
        note: 'Each clause corresponds to a value the property panel reports.',
      },
      {
        text: 'CNS-penetrant: TPSA under 90, MW under 450, at most one hydrogen-bond donor',
        note: 'Brain penetration requires a tighter physicochemical envelope than oral dosing.',
      },
    ],
  },

  {
    key: 'novelty',
    reach: 'full',
    name: 'Novelty and IP space',
    tag: 'distance from known compounds',

    definition: `**Novelty** is the structural distance between a proposed compound and the
                 compounds already known. It is measured as ECFP4 Tanimoto similarity to the
                 nearest known active, or by whether the Bemis-Murcko scaffold has been
                 reported. **Scaffold hopping** is replacement of the core while the
                 interaction pattern is retained.`,

    significance: `A generator trained on known actives reproduces known actives. A compound
                   covered by an existing claim cannot be developed independently of the
                   claim's holder. Scaffold hopping is also the standard response to a
                   liability that tracks the scaffold rather than the binding mode. The statins
                   are a common example of one pharmacophore realised on several scaffolds.`,

    methods: `Comparison against a compound bank. Above about 0.7 Tanimoto a compound belongs
              to the same series as its nearest neighbour. Below about 0.4 the
              structure-activity relationship of that series should not be assumed to
              transfer.`,

    here: [
      'Enable similarity in step 3. It is disabled by default because a fifth tool changes the routing cycle and breaks comparability with earlier runs.',
      'The subsystem searches ChEMBL and PubChem using a compound from the run, usually a generated one, rather than using the protein.',
      'Tanimoto is recomputed locally with ECFP4 for every hit rather than taken from the bank. The two banks report different quantities, and PubChem reports none.',
      'Select a safety mode. Strict rejects PAINS, withdrawn drugs and black-box-warning compounds; standard flags them; structural rules apply at every setting.',
    ],

    verdict: {
      read: `The nearest-neighbour distance together with the identity of the neighbour. A hit
             at Tanimoto 0.95 against a withdrawn drug is an informative result.`,
      fooled: `Reading a clean similarity search as freedom to operate. The banks contain
               published compounds. Patent claims cover structures that were never
               synthesised, so a compound absent from both banks may still fall inside one.`,
    },

    needs: ['similarity'],
    gap: `Similarity to a public bank is not a patent search, and no part of this framework
          reads patents.`,

    examples: [
      {
        text: 'novel chemotype: nothing above Tanimoto 0.5 to any known JAK2 active in ChEMBL',
        note: 'Checkable, because the similarity subsystem computes this value.',
      },
      {
        text:
          'scaffold-hop away from the pyrrolopyrimidine core while retaining the hinge-binding '
          + 'donor-acceptor pair',
        note: 'States what to retain as well as what to change.',
      },
    ],
  },

  {
    key: 'synth',
    reach: 'full',
    name: 'Synthesizability',
    tag: 'difficulty of preparation',

    definition: `**Synthetic accessibility** is an estimate of how difficult a compound is to
                 prepare. The SA score of Ertl and Schuffenhauer runs from 1 for readily
                 prepared compounds to 10 for difficult ones. It is derived from fragment
                 frequencies in known chemistry, with penalties for size, stereocentres and
                 unusual ring systems.`,

    significance: `Generative models optimise the objective supplied to them, and a docking
                   score contains no information about synthetic feasibility. The cost falls on
                   the design cycle. A programme able to prepare fifty analogues a year can map
                   a structure-activity relationship; one requiring fourteen steps per compound
                   prepares five and cannot.`,

    methods: `Retrosynthetic analysis establishes a route back to commercially available
              material, counts the steps, and checks that each disconnection corresponds to a
              known reaction. SA score substitutes for this when thousands of candidates must
              be ranked.`,

    here: [
      'Enable properties. SA score is computed with the rest of the panel.',
      'Filter before ranking. Ranking first allows an unpreparable compound to occupy the top position and set expectations for the run.',
      'In the Workbench, elaboration reports rejections per operator, which shows when crossover is failing on rigid seeds and mutation is supplying the quota.',
    ],

    verdict: {
      read: `SA below about 4 for any compound intended as a proposal, calibrated against the
             retrieved actives from the same run.`,
      fooled: `Accepting a low SA score as evidence of a route. The score is a
               fragment-frequency heuristic. It does not consider reagent availability and
               will pass compounds no chemist would attempt.`,
    },

    needs: ['properties'],
    gap: `Nothing performs retrosynthesis, checks commercial availability, or counts steps.`,

    examples: [
      {
        text:
          'nothing above SA score 4; a compound that cannot be prepared in a few steps from '
          + 'catalogue material does not count as a hit',
        note: 'Gives "preparable" a value, so the filter is applied rather than assumed.',
      },
    ],
  },

  {
    key: 'leadopt',
    reach: 'full',
    name: 'Lead optimisation',
    tag: 'improving a known active',

    definition: `**Lead optimisation** is the improvement of one property of a known active
                 compound while its other properties are held within acceptable bounds. It is
                 a multi-parameter optimisation problem. An **activity cliff** is a pair of
                 closely related compounds whose potencies differ by orders of magnitude.`,

    significance: `The properties are coupled. Increasing lipophilicity usually improves
                   binding, and simultaneously reduces solubility, increases metabolic
                   clearance and raises hERG risk. A series can improve in potency for a year
                   and produce compounds less suitable as drugs than its starting point.
                   Ligand efficiency was introduced to detect this; see §3. Activity cliffs are
                   a second hazard, and are not predicted by smooth models.`,

    methods: `Floors are fixed on the properties that must not degrade, one property is
              optimised, and efficiency metrics are tracked so that improvement cannot be
              obtained by adding molecular weight. Progress is assessed across a series rather
              than for a single compound.`,

    here: [
      'Enable generation, properties and docking. GraphGA elaborates by crossover and mutation from seeds already present in the run.',
      'Alternatively use the Workbench directly: supply a parent, elaborate it into analogues, dock the batch into a prepared pocket, and read the ranked table.',
      'Retain the parent in every comparison. It is the baseline, and a run that does not beat it is itself a result.',
      'A single seed produces mutations only. Crossover requires two parents and fails on rigid scaffolds, which the per-operator rejection counts make visible.',
    ],

    verdict: {
      read: `Ligand efficiency alongside the raw score, together with the floors that were
             set. A compound gaining 1 kcal/mol by gaining twelve heavy atoms has not
             improved.`,
      fooled: `Reporting the best compound in a large batch. The expected maximum improves
               with batch size independently of quality, as measured in §4. The median is the
               stable statistic.`,
    },

    needs: ['generation', 'properties', 'docking'],
    gap: `The generator operates on molecular graphs. It has no access to the series'
          structure-activity relationship, cannot recognise an activity cliff, and may
          re-propose a modification already tested.`,

    examples: [
      {
        text:
          'start from ruxolitinib: beat its docking score while keeping QED above 0.6 and SA '
          + 'below 4',
        note: 'One property to improve, the remainder held as floors.',
      },
      {
        text: 'reduce cLogP by one unit without losing more than 0.5 kcal/mol of docking score',
        note: 'States an exchange rate rather than a list of objectives.',
      },
    ],
  },

  {
    key: 'denovo',
    reach: 'manual',
    name: 'Hit finding from structure',
    tag: 'no known ligand',

    definition: `**Structure-based hit finding** is the identification of initial active
                 compounds for a target whose structure is known and for which no ligand has
                 been reported. The **hit rate** is the fraction of compounds ordered and
                 assayed that showed measurable activity.`,

    significance: `Drug-like chemical space is commonly estimated at around 10^60 compounds,
                   so enumeration is not possible. The starting point determines the accessible
                   chemistry, the liabilities inherited and the available patent space.
                   High-throughput screening obtains a starting point by brute force, at hit
                   rates typically below one percent.`,

    methods: `Virtual screening of a purchasable catalogue, fragment screening, or a generative
              model conditioned on the binding site. All three produce candidates that must
              then be obtained and assayed. Results are reported as a hit rate, which is the
              measurement that distinguishes a method from a plausible account.`,

    here: [
      'Open the Workbench and prepare a pocket from a PDB entry. If no ligand code is given it ranks the bound ligands, which shows what defines the site.',
      'Dock a batch and read the ranked table with the 3D viewer. A high score on an implausible pose is identifiable before the compound is proposed.',
      'Pocket-conditioned generation with DiffSBDD is installed under external/ and is not wired into a budgeted run.',
    ],

    verdict: {
      read: `The score distribution and the predicted pose, and eventually a hit rate from
             assayed compounds. Everything earlier is a hypothesis about what to test.`,
      fooled: `Quoting a generator's best score. That compound was selected using the same
               scoring function used to evaluate it.`,
    },

    needs: ['docking'],
    gap: `There is no routed de novo generator, so a budgeted run cannot perform this. The
          Workbench embeds, elaborates and docks interactively, under manual control.`,

    examples: [
      {
        text:
          'no known binders for this site: propose chemotypes that fit the pocket, then filter '
          + 'on PAINS and SA before ranking by score',
        note: 'Filter before ranking, so a reactive compound cannot win on score alone.',
      },
    ],
  },

  {
    key: 'resistance',
    reach: 'manual',
    name: 'Resistance and mutants',
    tag: 'the target changes',

    definition: `**Resistance** is the loss of activity that follows mutation of the target. A
                 **gatekeeper mutation** alters a residue at the entrance to the binding site,
                 preventing the inhibitor from binding while the protein remains functional.`,

    significance: `In oncology and infectious disease the target is under selection, so
                   resistance follows from effective treatment. Imatinib loses activity against
                   BCR-ABL carrying T315I, and ponatinib was designed with a carbon-carbon
                   triple bond to accommodate the larger residue. EGFR T790M has the same
                   effect on first-generation EGFR inhibitors, and osimertinib was developed
                   against the mutant form. Both are cited as demonstrations of
                   structure-guided design.`,

    methods: `Design against the mutant structure, then confirm activity against the wild type
              or establish that it is not required. In a resistant tumour the mutant is the
              form that matters. The problem is a selectivity problem between two forms of one
              protein.`,

    here: [
      'Obtain a structure of the mutant and use it as the target. Receptor preparation does not depend on why the residue differs.',
      'Run the wild type separately with identical settings and compare the traces manually, as for selectivity.',
      'Inspect the pose as well as the score. Steric clash with a larger gatekeeper is a geometric claim, visible in the 3D viewer.',
    ],

    verdict: {
      read: `A retained score against the mutant, with a pose that accounts for it. Successful
             design here normally has a visible structural explanation.`,
      fooled: `Docking into an unvalidated mutant structure. A homology model or an
               over-minimised pocket can accommodate compounds the real protein cannot, and
               the score does not indicate this.`,
    },

    needs: ['docking'],
    gap: `Nothing automates the wild-type comparison, and nothing checks that a mutant
          structure is physically reasonable.`,

    examples: [
      {
        text:
          'retain potency against the gatekeeper mutant; a compound that fits only the '
          + 'wild-type pocket is already obsolete',
        note: 'Pair with a second run against the wild-type structure.',
      },
    ],
  },
];

const RULES = [
  {
    head: 'State a failure condition',
    body: `A goal that cannot be disappointed gives a routing policy nothing to steer by, and
           gives the reader nothing to check the run against afterwards. "Potent inhibitors"
           has no failure condition. "Anything weaker than pChEMBL 7 is a miss" has one.`,
  },
  {
    head: 'Use numbers',
    body: `"Good oral bioavailability" cannot be checked against a result. "MW under 500,
           cLogP under 5, TPSA under 140" corresponds to three values the property panel
           reports directly.`,
  },
  {
    head: 'Name the exchange rate',
    body: `Goals with several objectives require a statement of what may be given up. A list
           of properties to maximise describes a preferred outcome. "Half a kcal/mol of
           docking score per log unit of lipophilicity" describes a decision rule.`,
  },
  {
    head: 'Record what would change the conclusion',
    body: `The result that would end the investigation should be written down before the run.
           A goal that cannot fail produces confirmation rather than evidence.`,
  },
  {
    head: 'Do not request unmeasurable quantities',
    body: `No subsystem measures half-life, clearance or a selectivity ratio. Including them
           in a goal produces a trace discussing quantities the run never computed.`,
  },
  {
    head: 'Fix the budget in advance',
    body: `Extending a search after seeing early results allows the stopping point to be
           chosen by the scoring function's noise. Arms being compared should share budget,
           seed and tool selection.`,
  },
];

/* --------------------------------------------------------------------------------------- */

/* `reach` is the editorial answer: how far a budgeted run gets on this kind of goal. `needs`
 * is mechanical: whether those subsystems are switched on. A goal can be unreachable with
 * every tool installed, as selectivity is, so the tools may only downgrade the verdict. */
const REACH = {
  full: { cls: 'ok', label: 'a run can pursue this' },
  partial: { cls: 'part', label: 'partly' },
  manual: { cls: 'none', label: 'not in a run' },
};

function support(goal, installed) {
  const missing = goal.needs.filter((k) => !installed.has(k));
  if (missing.length) {
    return {
      cls: 'none',
      label: missing.length === goal.needs.length ? 'tools switched off' : 'missing a tool',
    };
  }
  return REACH[goal.reach];
}

/* Minimal inline markup for the body text: **bold** for a defined term, which is the only
 * emphasis the article uses. Splitting on the delimiter rather than setting innerHTML keeps
 * the content as text nodes, so nothing in these strings can become markup by accident. */
function rich(text) {
  return text.split(/\*\*(.+?)\*\*/g).map((part, i) => (i % 2 ? html`<b>${part}</b>` : part));
}

const SECTIONS = [
  { id: 'overview', n: '1', title: 'Overview' },
  { id: 'types', n: '2', title: 'Kinds of goal', subs: GOALS },
  { id: 'numbers', n: '3', title: 'Reading the numbers' },
  { id: 'budget', n: '4', title: 'Budgets' },
  { id: 'writing', n: '5', title: 'Writing a goal' },
  { id: 'metrics', n: '6', title: 'Metrics glossary' },
];

/* Contents rail with a reading indicator. The active section is the last heading above the
 * reading line, not whichever heading happens to be visible: headings are short and sections
 * are long, so for most of a scroll no heading is on screen at all.
 *
 * A passive scroll listener. IntersectionObserver also works and was tried first; this needs
 * no rootMargin tuning to express "above the line", which is the actual question.
 *
 * Scroll events, IntersectionObserver callbacks and requestAnimationFrame are all delivered
 * as part of the rendering lifecycle, so a tab that is not painting receives none of them
 * while setTimeout continues to run. Measured here, a hidden tab received zero scroll events
 * across a programmatic scroll of fourteen thousand pixels. That needs no workaround, since
 * an indicator nobody can see does not need updating, but `pick` also runs on
 * visibilitychange so the rail is correct as soon as the tab is shown.
 */
function useActiveSection(ids) {
  const [active, setActive] = useState(ids[0]);

  useEffect(() => {
    const els = ids.map((id) => document.getElementById(id)).filter(Boolean);
    if (!els.length) return undefined;

    const LINE = 96; // just below the sticky app header
    const pick = () => {
      let cur = els[0].id;
      for (const el of els) {
        if (el.getBoundingClientRect().top > LINE) break;
        cur = el.id;
      }
      setActive(cur);
    };

    addEventListener('scroll', pick, { passive: true });
    addEventListener('resize', pick, { passive: true });
    document.addEventListener('visibilitychange', pick);
    pick();
    return () => {
      removeEventListener('scroll', pick);
      removeEventListener('resize', pick);
      document.removeEventListener('visibilitychange', pick);
    };
  }, [ids.join('|')]);

  return active;
}

/* In-page navigation, done explicitly rather than left to the browser.
 *
 * These links carry a real `href` so they can be copied, opened in a new tab and followed
 * with JavaScript disabled. But following one normally means the browser changes the hash,
 * fails to find an element named `/goals#g-potency`, and applies its own scroll handling for
 * a history entry it has just created. Measured here, that left every click in a sequence
 * short of its target by roughly the height of one section, while the same navigation
 * performed alone landed exactly. Rather than race it, the click is handled: the default is
 * prevented, the history entry is written, and the scroll is performed directly.
 *
 * The effect in `GoalsView` still handles a cold load, where there is no click to intercept.
 */
function goToAnchor(e, id) {
  if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return; // let the browser open a tab
  e.preventDefault();
  const url = '#/goals#' + id;
  if (location.hash !== url) history.pushState({ drugorchAnchor: true }, '', url);
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ block: 'start' });
  // pushState does not fire hashchange, so tell the router itself.
  dispatchEvent(new HashChangeEvent('hashchange'));
}


function Heading({ level, id, n, children }) {
  const Tag = `h${level}`;
  return html`
    <${Tag} id=${id} class="wh">
      ${n && html`<span class="wh-n">${n}</span>`}
      <span class="wh-t">${children}</span>
      <a class="wh-link" href=${'#/goals#' + id} title="Link to this section"
         onClick=${(e) => goToAnchor(e, id)} aria-label="Link to this section">¶</a>
    <//>
  `;
}

/* The glossary, as the article's reference section.
 *
 * Entries are served from `/api/glossary`, the same source the popovers in the rest of the
 * interface read, so a metric cannot be defined one way in a tooltip and another here.
 *
 * Each entry is an anchor target, so `#/goals#m-ligand_efficiency` is a link a colleague
 * can be sent, and the `^` beside each heading returns to where the reader came from. That is
 * the footnote convention, and it is the right one *inside a document*: the reader is here
 * to read, so moving them is not disruptive. Elsewhere in the application the same entries
 * appear as popovers instead, because moving someone away from a table of numbers to
 * explain one of its columns loses the thing they were looking at.
 */
function Glossary({ data, error, current }) {
  if (error) return html`<p class="note err">Could not load the glossary: ${error}</p>`;
  if (!data) return html`<div class="empty">loading…</div>`;

  const byGroup = new Map();
  for (const m of data.metrics) {
    if (!byGroup.has(m.group)) byGroup.set(m.group, []);
    byGroup.get(m.group).push(m);
  }

  const back = (e) => {
    /* Only go back to a jump this page actually made.
     *
     * `history.length > 1` counts the whole tab session, so a reader who simply scrolled
     * down to the glossary, or who arrived from another site and pasted a deep link, was
     * sent to the previous page: out of the article, often out of the application. The
     * marker is written by `goToAnchor`, so it is present exactly when there is somewhere
     * within the article to return to. Otherwise the link behaves as written and goes to
     * the top of the glossary. */
    if (history.state?.drugorchAnchor) {
      e.preventDefault();
      history.back();
    }
  };

  return html`
    <p>
      Every quantity the interface displays, what produces it, and how to read it. The same
      entries appear as popovers on the column headings elsewhere in the application.
    </p>
    ${[...byGroup.entries()].map(
      ([group, metrics]) => html`
        <section key=${group}>
          <h4 class="wsub">${data.groups[group] || group}</h4>
          <dl class="gloss">
            ${metrics.map(
              (m) => html`
                <div class=${'gentry' + (current === 'm-' + m.key ? ' here' : '')}
                     id=${'m-' + m.key} key=${m.key}>
                  <dt>
                    ${m.term}
                    ${m.direction !== 'context' &&
                    html`<span class=${'gdir ' + m.direction}>
                      ${m.direction === 'higher' ? '↑ higher is better' : '↓ lower is better'}
                    </span>`}
                    <a class="gback" href="#/goals#metrics" onClick=${back}
                       title="Back to where you were">^</a>
                  </dt>
                  <dd>
                    <p>${m.short}</p>
                    ${m.range && html`<p class="gmeta"><b>Range.</b> ${m.range}</p>`}
                    ${m.threshold && html`<p class="gmeta"><b>Threshold.</b> ${m.threshold}</p>`}
                    ${m.computed && html`<p class="gmeta"><b>Computed.</b> ${m.computed}</p>`}
                    <p class="gmeta"><b>Reading.</b> ${m.reading}</p>
                    ${m.caution && html`<p class="gcaution"><b>Caution.</b> ${m.caution}</p>`}
                    ${m.see_also?.length > 0 &&
                    html`<p class="gmeta small muted">
                      See also:${' '}
                      ${m.see_also.map((k, i) => {
                        const other = data.metrics.find((x) => x.key === k);
                        return html`<span key=${k}>${i ? ', ' : ''}<a href=${'#/goals#m-' + k}
                          onClick=${(e) => goToAnchor(e, 'm-' + k)}
                          >${other ? other.term : k}</a></span>`;
                      })}
                    </p>`}
                  </dd>
                </div>
              `,
            )}
          </dl>
        </section>
      `,
    )}
  `;
}


export function GoalsView({ manifest, anchor, onUseGoal }) {
  const installed = new Set((manifest?.subsystems || []).map((s) => s.key));
  const [copied, setCopied] = useState(null);

  const ids = SECTIONS.flatMap((s) => [s.id, ...(s.subs || []).map((g) => 'g-' + g.key)]);
  const active = useActiveSection(ids);
  const [glossary, setGlossary] = useState(null);
  const [glossaryError, setGlossaryError] = useState(null);

  useEffect(() => {
    let live = true;
    loadGlossary()
      .then((d) => live && setGlossary(d))
      .catch((e) => live && setGlossaryError(e.message));
    return () => {
      live = false;
    };
  }, []);

  /* Scroll to the element the URL names. The browser cannot do this itself: the fragment is
   * the part after the second `#`, so there is no element whose id matches the real hash.
   *
   * `glossary` is a dependency rather than decoration. Glossary entries are the most linked
   * targets and they do not exist until the fetch resolves, so an effect keyed on `anchor`
   * alone finds nothing and silently does nothing. Re-running when the data lands is what
   * makes a pasted `#/goals#m-qed` work on a cold load. */
  useEffect(() => {
    if (!anchor) return;
    const el = document.getElementById(anchor);
    if (el) el.scrollIntoView({ block: 'start' });
  }, [anchor, glossary]);

  useEffect(() => {
    if (copied == null) return undefined;
    const t = setTimeout(() => setCopied(null), 1400);
    return () => clearTimeout(t);
  }, [copied]);

  function use(text) {
    setCopied(text);
    try {
      sessionStorage.setItem(GOAL_HANDOFF, text);
    } catch {
      /* private mode: fall through to the callback, which does not need storage */
    }
    onUseGoal?.(text);
  }

  return html`
    <div class="wiki">
      <nav class="wiki-toc" aria-label="Contents">
        <h2>Contents</h2>
        <ol>
          ${SECTIONS.map(
            (s) => html`
              <li key=${s.id}>
                <a class=${active === s.id ? 'cur' : ''} href=${'#/goals#' + s.id}
                   onClick=${(e) => goToAnchor(e, s.id)}>
                  <span class="tn">${s.n}</span>${s.title}
                </a>
                ${s.subs &&
                html`<ol>
                  ${s.subs.map(
                    (g, i) => html`
                      <li key=${g.key}>
                        <a class=${active === 'g-' + g.key ? 'cur' : ''}
                           href=${'#/goals#g-' + g.key}
                           onClick=${(e) => goToAnchor(e, 'g-' + g.key)}>
                          <span class="tn">${s.n}.${i + 1}</span>${g.name}
                        </a>
                      </li>
                    `,
                  )}
                </ol>`}
              </li>
            `,
          )}
        </ol>
      </nav>

      <article class="wiki-body">
        <h1 class="wiki-title">Goals in drug discovery</h1>
        <p class="wiki-sub">
          Reference for the goal field: what it controls, the kinds of goal a discovery
          programme has, and how to evaluate a result against one.
        </p>

        <p class="hatnote">
          Only the <b>llm</b> routing policy reads the goal. The other five policies (fixed,
          random, roundrobin, scripted and oracle) follow fixed rules and never receive this
          text. On those runs the goal is a label on the record, which is what makes them
          controls.
        </p>

        <${Heading} level="2" id="overview" n="1">Overview<//>
        <p>
          A <b>goal</b> is a free-text statement of what would count as success in a run. It is
          included in the prompt sent to the routing model, so it affects which subsystem is
          called next and which thresholds are chosen. It has no other effect. The values that
          filter compounds are those a policy passes to a subsystem, and they appear in the
          trace.
        </p>
        <p>
          Section 2 describes eight recurring kinds of goal. Each entry gives a definition, the
          reason the problem matters, how it is approached in practice, the procedure available
          here, how to evaluate a result, and what this framework cannot do. Section 3 covers
          the conversions needed to read the result panels. Section 4 explains why runs are
          budgeted. Section 5 lists conventions for writing a goal.
        </p>

        <${Heading} level="2" id="types" n="2">Kinds of goal<//>
        <p>
          Each entry carries a status box stating how far a budgeted run can pursue that goal.
          The status is downgraded automatically when a required subsystem is disabled. Most of
          these goals are only partly reachable here, and the limitations are listed with each
          entry.
        </p>

        ${GOALS.map((g, i) => {
          const s = support(g, installed);
          return html`
            <section class="goalsec" key=${g.key}>
              <${Heading} level="3" id=${'g-' + g.key} n=${'2.' + (i + 1)}>${g.name}<//>

              <aside class="infobox">
                <div class="ib-head">${g.name}</div>
                <dl>
                  <dt>Concerns</dt>
                  <dd>${g.tag}</dd>
                  <dt>Status</dt>
                  <dd><span class=${'sup ' + s.cls}>${s.label}</span></dd>
                  <dt>Subsystems</dt>
                  <dd>
                    ${g.needs.map(
                      (k) => html`<span
                        key=${k}
                        class=${'chip' + (installed.has(k) ? '' : ' off')}
                        title=${installed.has(k) ? 'installed' : 'not installed'}
                        >${k}</span
                      >`,
                    )}
                  </dd>
                </dl>
              </aside>

              <p>${rich(g.definition)}</p>

              <h4 class="wsub">Significance</h4>
              <p>${rich(g.significance)}</p>

              <h4 class="wsub">Methods</h4>
              <p>${rich(g.methods)}</p>

              <h4 class="wsub">Procedure</h4>
              <ol class="wsteps">
                ${g.here.map((step, j) => html`<li key=${j}>${step}</li>`)}
              </ol>

              <h4 class="wsub">Evaluation</h4>
              <dl class="judge">
                <dt class="good">Indicator</dt>
                <dd>${g.verdict.read}</dd>
                <dt class="bad">Common error</dt>
                <dd>${g.verdict.fooled}</dd>
              </dl>

              <p class="ambox"><b>Limitations.</b> ${g.gap}</p>

              <h4 class="wsub">Example goals</h4>
              ${g.examples.map(
                (ex) => html`
                  <div class="ex" key=${ex.text}>
                    <div class="ex-text">${ex.text}</div>
                    <div class="ex-note small muted">${ex.note}</div>
                    <button type="button" class="btn ghost sm" onClick=${() => use(ex.text)}>
                      ${copied === ex.text ? 'sent to setup' : 'Use this goal'}
                    </button>
                  </div>
                `,
              )}
            </section>
          `;
        })}

        <${Heading} level="2" id="numbers" n="3">Reading the numbers<//>
        <p>
          Result panels report energies, counts and ratios. The conversions below relate them
          to properties of compounds.
        </p>

        <h4 class="wsub">Affinity and free energy</h4>
        <p>
          Binding affinity and binding free energy are equivalent. At 298 K, <i>RT</i>·ln(10)
          equals <b>1.364 kcal/mol</b>, so a difference of 1.364 kcal/mol corresponds to a
          factor of ten in affinity, and 2.7 kcal/mol to a factor of a hundred.
        </p>
        <p>
          Applied in reverse, a published docking error of about 2 kcal/mol corresponds to a
          factor of roughly <b>29</b> in predicted affinity. Differences below about 1
          kcal/mol, a factor of 5, fall within that error.
        </p>

        <h4 class="wsub">Ligand efficiency</h4>
        <p>
          <b>Ligand efficiency</b> (LE) is binding free energy divided by heavy-atom count:
          LE = −ΔG / HAC. About 0.3 kcal/mol per atom is a common minimum. Raw binding energy
          increases with size, because additional atoms make additional contacts, and LE
          removes that dependence.
        </p>
        <p>
          LE is defined on experimental free energy. Computing it from a docking score
          reproduces the size dependence the metric was introduced to remove. From a measured
          affinity, LE = 1.364 × pChEMBL / HAC.
        </p>
        <p>Three measured JAK2 inhibitors:</p>
        <table class="wikitable">
          <thead>
            <tr>
              <th>compound</th><th>assay</th>
              <th class="num">pChEMBL</th><th class="num">heavy atoms</th>
              <th class="num">LE</th><th class="num">cLogP</th>
            </tr>
          </thead>
          <tbody>
            ${LE_ROWS.map(
              (r) => html`<tr key=${r.id}>
                <td class="mono">
                  ${r.id}${r.note && html`<span class="muted"> (${r.note})</span>`}
                </td>
                <td>${r.assay}</td>
                <td class="num mono">${r.pchembl.toFixed(2)}</td>
                <td class="num mono">${r.hac}</td>
                <td class="num mono"><b>${r.le.toFixed(3)}</b></td>
                <td class="num mono">${r.logp.toFixed(2)}</td>
              </tr>`,
            )}
          </tbody>
        </table>
        <p>
          The second and third compounds were measured in the same assay type at effectively
          the same potency, 7.24 against 7.28, a difference of 0.04 log units. One has 20 heavy
          atoms and the other 39. The additional 19 atoms produced no measurable gain in
          affinity and halved the ligand efficiency.
        </p>
        <p>
          Across all ${LE_STATS.n} measured actives for ${LE_STATS.target}, heavy-atom count
          correlates with potency at <b>r = ${LE_STATS.corrSizePotency.toFixed(2)}</b> and with
          ligand efficiency at <b>r = ${LE_STATS.corrSizeLe.toFixed(2)}</b>. Median heavy-atom
          count is ${LE_STATS.medianHac} (range ${LE_STATS.hacRange}), median LE
          ${' '}${LE_STATS.medianLe.toFixed(3)}, median pChEMBL
          ${' '}${LE_STATS.medianP.toFixed(2)}.
        </p>
        <p class="refnote"><b>Source.</b> ${PROVENANCE}</p>

        <h4 class="wsub">Lipophilic ligand efficiency</h4>
        <p>
          <b>Lipophilic ligand efficiency</b> (LLE) is pActivity minus cLogP, with 5 a common
          target. It detects potency obtained by increasing lipophilicity. LLE requires a
          measured activity, so it can be computed for retrieved compounds and not for
          generated ones.
        </p>

        <${Heading} level="2" id="budget" n="4">Budgets<//>
        <p>
          A <b>budget</b> limits the number of subsystem calls a run may make. Running a
          campaign has direct costs: model tokens when the policy is <b>llm</b>, GPU time, and
          requests to the public ChEMBL and PubChem endpoints. The budget exists for a
          different reason. It is a statistical constraint.
        </p>
        <p>
          Scoring functions carry error. Selecting the best compound by measured score finds
          both compounds that are genuinely good and compounds whose error is favourable. The
          second effect grows with the number of compounds screened. The first is bounded by
          the chemistry present in the pool. Beyond some point, further sampling locates errors
          in the scoring function rather than better compounds.
        </p>
        <${SelectionFigure} />
        <p>
          Two runs are therefore comparable only at equal budget, since an arm that searched
          ten times longer reports a better score without being better. The Workbench has no
          budget and creates no run, because its output is exploratory. The orchestrator has
          one, because its output is intended as evidence.
        </p>

        <${Heading} level="2" id="writing" n="5">Writing a goal<//>
        <dl class="wrules">
          ${RULES.map(
            (r) => html`
              <div key=${r.head}>
                <dt>${r.head}</dt>
                <dd>${r.body}</dd>
              </div>
            `,
          )}
        </dl>

        <${Heading} level="2" id="metrics" n="6">Metrics glossary<//>
        <${Glossary} data=${glossary} error=${glossaryError} current=${anchor} />
      </article>
    </div>
  `;
}
