# NFCorpus source and terms

The existing `corpus.json`, `queries.json` and `qrels.json` are derived from
NFCorpus by Vera Boteva, Demian Gholipour, Artem Sokolov and Stefan Riezler
(2016), via the BEIR Hugging Face distribution. Documents originate from the
biomedical/nutrition collection described on the
[original project page](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/).
The preparation script converts source fields to local JSON and selects the
323 test queries with relevance judgments; it retains 3,633 documents.

The repository's MIT code license does not relicense these data. The
[BEIR dataset card](https://huggingface.co/datasets/BeIR/nfcorpus/blob/main/README.md)
labels its distribution CC BY-SA 4.0. The original project's Terms of Use
describe free academic use and direct other uses of NutritionFacts data to
the applicable terms and authors. These notices describe different source
contexts; this repository does not resolve their scope into a new blanket
permission. Consult both sources for the intended use.

Unlike the newer GSM8K/HotpotQA pipeline, the historical
`download_nfcorpus.py` uses mutable dataset-server and `main` URLs and did not
record a pinned download revision or source receipt. The checked-in snapshot
is available for reproducing the historical experiment, but its exact upstream
download revision is not established. This limitation remains visible; no new
download, content change or license reassignment is implied by this note.
