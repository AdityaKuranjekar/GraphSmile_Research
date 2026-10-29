# p1b: data provenance and label check (no training changes, no GPU)

**Decision: kept as a finding. The IEMOCAP-6 feature file used by every run is byte-identical to the README release. No
differences found for IEMOCAP-6. IEMOCAP-4, MELD and CMU-MOSEI files on the server were not inspected.**

## Method
- Source A (experiments): `/DATA/rashmi_2521cs34/data/IEMOCAP.pkl`, the path set in the server's `run.py` (`IEMOCAP_path`, line 156,
  seen earlier in this work). Inspected on the server with `inspect_pkl.py`.
- Source B (README release): Google Drive folder from `README.md`
  (`drive.google.com/drive/folders/1l_ex1wnAAMpEtO71rjjM1MKC7W_olEVi`). The folder listing was readable and lists 4 files. Only
  `iemocap_multi_features.pkl` and `iemocap_multi_features_4.pkl` were downloaded (approved by the user) into a new scratch dir.
  The Baidu link was not used.
- Both files were read with a restricted unpickler that allows only numpy, collections, copyreg and codecs globals, never
  `pickle.load`. No blocked global was hit, so both files contain only plain containers and numpy arrays.
- `inspect_pkl.py` is kept untracked and is not part of this commit.

## 1. File identity, IEMOCAP-6
| Property | Experiments (server) | README release | Same? |
|---|---|---|---|
| File name | `IEMOCAP.pkl` | `iemocap_multi_features.pkl` | renamed only |
| Size (bytes) | 191,613,263 | 191,613,263 | yes |
| md5 | bfb8f1545d14ea2e1d9e1825380062b1 | bfb8f1545d14ea2e1d9e1825380062b1 | **yes, byte-identical** |

## 2. Structure, IEMOCAP-6 (identical in both)
Top-level: a `list` of 12 elements: IDs, Speakers, Labels, Text0, Text1, Text2, Text3, Audio, Visual, Sentence (each a dict
of 151 dialogues), then trainVid (list, 120) and testVid (list, 31). This matches the 12-tuple unpacked by `IEMOCAPDataset_BERT`
in `dataloader.py`. Train and test dialogues do not overlap; every dialogue has labels; no feature/label length mismatch.
The first train dialogue is `Ses02F_script01_3` in both, with identical per-modality array md5s (computed, equal).

## 3. Dimensions and class counts, IEMOCAP-6 (identical in both)
| Modality | Dimension | Expected | Same as expected? |
|---|---|---|---|
| Text0, Text1, Text2, Text3 | 1024 each | 1024 | yes |
| Visual | 342 | 342 | yes |
| Audio | 1582 | 1582 | yes |

`run.py` uses `embedding_dims = [1024, 342, 1582]`, consistent.

| Label index | 0 | 1 | 2 | 3 | 4 | 5 | Total |
|---|---|---|---|---|---|---|---|
| Train+val utterances | 504 | 839 | 1324 | 933 | 742 | 1468 | 5810 |
| Test utterances | 144 | 245 | 384 | 170 | 299 | 381 | 1623 |
| Paper order (Table IV) | Happy | Sad | Neutral | Angry | Excited | Frustrated | |

Totals match paper Table II (120 / 31 dialogues, 5,810 / 1,623 utterances).

## 4. Label order check (0 = Happy, 4 = Excited)
What the evidence supports, and what it does not:
- The pickle stores integer labels only, with no names, so the names cannot be read from the file.
- The row sums of the confusion matrix in `results/logs/IEMOCAP_full.log` are 144 for index 0 and 299 for index 4, equal to the
  test counts above. So the index order our runs report is the order of this file.
- `dataloader.py` maps labels 1, 3, 5 to negative, 2 to neutral, and 0, 4 to positive. Paper Table I says negative =
  Sad, Angry, Frustrated; neutral = Neutral; positive = Happy, Excited. With Table IV's column order (Happy, Sad, Neutral,
  Angry, Excited, Frustrated) that gives exactly 0 = Happy, 1 = Sad, 2 = Neutral, 3 = Angry, 4 = Excited, 5 = Frustrated.
- Class sizes rise and fall in that order (smallest 0 and 4, largest 5 and 2), which fits Happy and Excited being the minority classes.
- Not verified: an independent source stating the index-to-name map. The authors' code was not read for this; the evidence above is
  internal consistency, not a documented mapping. It is strong but indirect.

## 5. IEMOCAP-4, README release only
Server copy not inspected, so no comparison is possible. README file (`iemocap_multi_features_4.pkl`, 30,653,738 bytes,
md5 39498a707e4db62d09089e168fe7ea6e): 9-element list; 120 train+val and 31 test dialogues; Text 1024, Visual 512, Audio 100
(`run.py` expects [1024, 512, 100]); 3,600 and 943 utterances (match Table II); per-class train+val 504, 839, 1324, 933 and
test 144, 245, 384, 170. Excited is excluded here (Happy has 504 = the 6-class Happy count), as Table I (positive = Happy only) says.

## Differences found
| Item | Experiments | README release | Difference |
|---|---|---|---|
| IEMOCAP-6 file md5 / size | bfb8f154..., 191,613,263 | bfb8f154..., 191,613,263 | none |
| Structure, dims, class counts, split | as above | as above | none |
| Drive folder title | n/a | "CFN-ESA" | The folder is titled with a different project name. Not verified that it is the authors' official release. |
| Drive modified dates | n/a | iemocap_multi_features.pkl 19 Jan 2025 | irrelevant to content; bytes match |

## What this means
- For IEMOCAP-6, data provenance is **not** the cause of the 4.5-point WF1 or the Happy gap to the paper: our features are exactly the
  released ones. Remaining untested: whether the released file equals the one the authors used for the paper (cannot be checked from here).
- No code was changed and no GPU job was run. The other three server pickles should be hashed the same way if you want the
  full table closed.
