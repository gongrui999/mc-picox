"""
Convert raw EBM-NLP 2.00 data to BioC XML format required by step_0_preprocess.py.

Reads tokens, POS tags, and aggregated P/I/O annotations from the raw EBM-NLP
directory and produces a sentence-split BioC XML file.

Usage:
    python convert_ebm_nlp_to_bioc.py \
        --ebm_nlp_dir data/raw/ebm_nlp_2_00 \
        --output_file data/bioc/input/ebm_nlp_2_00_ssplit.xml
"""

import argparse
import os
from datetime import date

from bioc import biocxml, BioCCollection, BioCDocument, BioCPassage, BioCSentence, BioCAnnotation, BioCLocation


SENT_END_TOKENS = {'.', '!', '?'}


def load_tokens(pmid: str, doc_dir: str):
    path = os.path.join(doc_dir, f"{pmid}.tokens")
    with open(path, encoding="utf-8") as f:
        return [t for t in f.read().split('\n') if t]


def load_pos(pmid: str, doc_dir: str):
    path = os.path.join(doc_dir, f"{pmid}.pos")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [t for t in f.read().split('\n') if t]


def load_ann(pmid: str, ann_base: str, pico: str, split: str):
    """Load aggregated annotation for a given PMID and PICO type."""
    for subdir in [split, os.path.join(split, "gold")]:
        path = os.path.join(ann_base, pico, subdir, f"{pmid}.AGGREGATED.ann")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return [int(x) for x in f.read().split() if x.strip()]
    return None


def split_into_sentences(tokens):
    """Split a flat token list into sentences on sentence-ending punctuation."""
    sentences = []
    current = []
    for tok in tokens:
        current.append(tok)
        if tok in SENT_END_TOKENS:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def build_document(pmid, tokens, pos_tags, p_labels, i_labels, o_labels):
    doc = BioCDocument()
    doc.id = pmid

    passage = BioCPassage()
    passage.offset = 0
    passage.text = " ".join(tokens)

    sentences = split_into_sentences(tokens)
    token_idx = 0
    char_offset = 0

    for sent_tokens in sentences:
        sent = BioCSentence()
        sent.offset = char_offset
        sent.text = " ".join(sent_tokens)

        for j, tok in enumerate(sent_tokens):
            ann = BioCAnnotation()
            ann.id = str(token_idx)
            ann.text = tok

            pos = pos_tags[token_idx] if pos_tags and token_idx < len(pos_tags) else "NN"
            ann.infons["POS"] = pos

            p = str(p_labels[token_idx]) if p_labels and token_idx < len(p_labels) else "0"
            i = str(i_labels[token_idx]) if i_labels and token_idx < len(i_labels) else "0"
            o = str(o_labels[token_idx]) if o_labels and token_idx < len(o_labels) else "0"
            ann.infons["starting_spans/participants"] = p
            ann.infons["starting_spans/interventions"] = i
            ann.infons["starting_spans/outcomes"] = o
            # hierarchical labels default to 0
            ann.infons["hierarchical_labels/participants"] = "0"
            ann.infons["hierarchical_labels/interventions"] = "0"
            ann.infons["hierarchical_labels/outcomes"] = "0"

            tok_char_offset = passage.text.index(tok, char_offset - passage.offset)
            ann.add_location(BioCLocation(tok_char_offset, len(tok)))

            sent.add_annotation(ann)
            token_idx += 1
            char_offset += len(tok) + 1  # +1 for space

        passage.add_sentence(sent)

    doc.add_passage(passage)
    return doc


def get_pmid_split(ann_base, pico):
    """Get {pmid: split} mapping from annotation directory."""
    result = {}
    for split in ["train", os.path.join("test", "gold")]:
        split_dir = os.path.join(ann_base, pico, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in os.listdir(split_dir):
            if fname.endswith(".AGGREGATED.ann"):
                pmid = fname.replace(".AGGREGATED.ann", "")
                result[pmid] = "train" if split == "train" else "test"
    return result


def main():
    parser = argparse.ArgumentParser(description="Convert EBM-NLP to BioC XML")
    parser.add_argument("--ebm_nlp_dir", type=str,
                        default=r"D:\PICOX\EBM-NLP\ebm_nlp_2_00\ebm_nlp_2_00",
                        help="Path to extracted ebm_nlp_2_00 directory")
    parser.add_argument("--output_file", type=str,
                        default="data/bioc/input/ebm_nlp_2_00_ssplit.xml",
                        help="Output BioC XML path")
    args = parser.parse_args()

    doc_dir = os.path.join(args.ebm_nlp_dir, "documents")
    ann_base = os.path.join(args.ebm_nlp_dir, "annotations", "aggregated", "starting_spans")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Collect all PMIDs that have annotations for ALL three PICO types
    p_pmids = get_pmid_split(ann_base, "participants")
    i_pmids = get_pmid_split(ann_base, "interventions")
    o_pmids = get_pmid_split(ann_base, "outcomes")
    all_pmids = set(p_pmids) & set(i_pmids) & set(o_pmids)
    print(f"Found {len(all_pmids)} PMIDs with complete P/I/O annotations")

    collection = BioCCollection()
    collection.source = "EBM-NLP"
    collection.date = str(date.today())
    collection.key = ""

    skipped = 0
    for idx, pmid in enumerate(sorted(all_pmids)):
        if (idx + 1) % 500 == 0:
            print(f"  Processing {idx+1}/{len(all_pmids)}...")

        tokens_path = os.path.join(doc_dir, f"{pmid}.tokens")
        if not os.path.exists(tokens_path):
            skipped += 1
            continue

        tokens = load_tokens(pmid, doc_dir)
        pos_tags = load_pos(pmid, doc_dir)

        split = p_pmids.get(pmid, "train")
        p_labels = load_ann(pmid, ann_base, "participants", split)
        i_labels = load_ann(pmid, ann_base, "interventions", split)
        o_labels = load_ann(pmid, ann_base, "outcomes", split)

        if not tokens:
            skipped += 1
            continue

        # Pad/truncate labels to match token count
        n = len(tokens)
        for labels in [p_labels, i_labels, o_labels]:
            if labels is not None:
                while len(labels) < n:
                    labels.append(0)

        doc = build_document(pmid, tokens, pos_tags, p_labels, i_labels, o_labels)
        collection.add_document(doc)

    print(f"Built collection with {len(collection.documents)} documents ({skipped} skipped)")
    print(f"Writing to {args.output_file} ...")

    with open(args.output_file, "w", encoding="utf-8") as fp:
        biocxml.dump(collection, fp)

    print("Done.")


if __name__ == "__main__":
    main()
