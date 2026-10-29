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

---

# Addendum: speaker, session and sentiment checks (pass 1b as re-issued)
Computed on the README file with the restricted reader. The server `IEMOCAP.pkl` is byte-identical (same md5), so every number
below holds for the file the experiments used. The sentiment check ran the project's own `IEMOCAPDataset_BERT` with its
`pickle.load` redirected to the restricted reader.

## A. Table II check (paper vs file), IEMOCAP-6
| Quantity | Paper Table II | File | Same? |
|---|---|---|---|
| Dialogues train+val / test | 120 / 31 | 120 / 31 | yes |
| Utterances train+val / test | 5,810 / 1,623 | 5,810 / 1,623 | yes |
IEMOCAP-4 (README file): 120 / 31 and 3,600 / 943 also match Table II. MELD and CMU-MOSEI: not downloaded, not checked.

## B. Utterances per class and per speaker role (F = female, M = male speaker role), IEMOCAP-6
Train+val (M 3,067, F 2,743):
| Class (index) | Happy (0) | Sad (1) | Neutral (2) | Angry (3) | Excited (4) | Frustrated (5) | Total |
|---|---|---|---|---|---|---|---|
| F | 275 | 425 | 512 | 511 | 361 | 659 | 2,743 |
| M | 229 | 414 | 812 | 422 | 381 | 809 | 3,067 |
| All | 504 | 839 | 1,324 | 933 | 742 | 1,468 | 5,810 |

Test (F 751, M 872):
| Class (index) | Happy (0) | Sad (1) | Neutral (2) | Angry (3) | Excited (4) | Frustrated (5) | Total |
|---|---|---|---|---|---|---|---|
| F | 77 | 132 | 221 | 78 | 82 | 161 | 751 |
| M | 67 | 113 | 163 | 92 | 217 | 220 | 872 |
| All | 144 | 245 | 384 | 170 | 299 | 381 | 1,623 |

Note: in test, Excited is 217 male vs 82 female utterances (73% male), while train+val Excited is 381 male vs 361 female
(51% male). Not tested whether this shift matters for the Happy/Excited confusion.

## C. Sessions and actors
- Test dialogues are all from Session 5 (31 of 31); train+val dialogues are Sessions 1 to 4 (28, 30, 32, 30). So the test set is
  speaker-independent from train, while any random split of train+val is not.
- Utterance IDs encode session and speaker letter; the letter equals the speaker role for all 5,810 + 1,623 utterances. Utterances
  per session and role, train+val: Ses01 F 641 / M 732, Ses02 F 654 / M 702, Ses03 F 732 / M 837, Ses04 F 716 / M 796; test: Ses05 F 751 / M 872.
- Our validation split (first 12 dialogues of `trainVid`, used by `--clean_valid`): 664 utterances from Sessions 1 to 4
  (Ses01 2, Ses02 6, Ses03 2, Ses04 2 dialogues), with class counts 45, 93, 163, 79, 166, 118 (Happy only 45). It shares actors
  with the training dialogues, so it is a speaker-dependent split, and Happy F1 on it rests on 45 utterances.

## D. Sentiment labels (from the project's own dataloader)
Emotion to sentiment, counted from `labels_sentiment` (0 = negative, 1 = neutral, 2 = positive):
| Emotion (index) | Train+val sentiment | Test sentiment |
|---|---|---|
| Happy (0) | positive: 504 of 504 | positive: 144 of 144 |
| Sad (1) | negative: 839 of 839 | negative: 245 of 245 |
| Neutral (2) | neutral: 1,324 of 1,324 | neutral: 384 of 384 |
| Angry (3) | negative: 933 of 933 | negative: 170 of 170 |
| Excited (4) | positive: 742 of 742 | positive: 299 of 299 |
| Frustrated (5) | negative: 1,468 of 1,468 | negative: 381 of 381 |
Happy and Excited both map to Positive, and Sad, Angry, Frustrated to Negative, as in paper Table I. This confirms the label to
name assignment only through the code's own mapping (see section 4 above); the pickle itself has no names.

## E. Feature files not yet checked (what to download and compare)
The server has other pickles (MELD, CMU-MOSEI, IEMOCAP-4 paths in `run.py`) that were not hashed. From the README Drive folder:
`iemocap_multi_features_4.pkl` (downloaded, md5 39498a70...), `meld_multi_features.pkl` (269.1 MB, not downloaded),
`cmumosei_multi_regression_features.pkl` (169.6 MB, not downloaded). To close the table, run `md5sum` on the server copies of those
three paths and compare with the README files; the MELD and CMU-MOSEI downloads need your approval.

## Differences found (updated)
| Item | Result |
|---|---|
| IEMOCAP-6 file, structure, dims, per-class counts, Table II counts | identical to the README release and to the paper |
| Per-speaker counts | no source to compare against (paper gives none); reported above |
| Label order and sentiment mapping | consistent with Table I and Table IV; no names in the pickle |
| IEMOCAP-4 / MELD / CMU-MOSEI server files | not compared (see E) |

---

# Addendum 2: MELD and CMU-MOSEI README files (approved download, restricted reader)
Files downloaded from the README Drive folder into a new scratch dir (not in the repo): `meld_multi_features.pkl`
(282,181,201 bytes, md5 87ec0e914aa34429587d9396bf3db42a) and `cmumosei_multi_regression_features.pkl` (177,842,139 bytes,
md5 b36414e92410407c86360d69c9af827f); sizes equal the Drive listing (269.1 and 169.6 MiB). Read with the restricted reader.
The first pass blocked CMU-MOSEI on `builtins.list`; the allow-list was widened to the plain container builtins
(list, dict, tuple, set, frozenset, slice, range, complex, bytearray) and nothing else. No code-executing global was requested by either file.
Labels and sentiment counts below come from the project's own `MELDDataset_BERT` and `CMUMOSEIDataset7`, run with the restricted reader.

**The server copies of these two files (and IEMOCAP-4) have not been hashed yet, so there is no README-vs-experiments comparison for them.**
The comparison in this section is README file vs paper Table II.

## F. Structure
| Property | MELD | CMU-MOSEI |
|---|---|---|
| Top-level | list of 14: IDs, Speakers, Labels, Sentiments, Text0 to Text3, Audio, Visual, Sentence, trainVid (set), testVid (set), None | list of 9: IDs, Speakers, Labels, Text, Audio, Visual, Sentence, trainVid (array), testVid (array) |
| Dialogue entries | 1,432 | 3,288 (Labels dict has 3,290 keys) |
| Dims | text 1024 (x4), audio 300, visual 342, speaker one-hot 9 | text 1024, audio 384, visual 35 |
| `run.py` expects | [1024, 342, 300] | [1024, 35, 384] |
| Matches `run.py`? | yes | yes |
| Train/test overlap, length mismatches | 0, 0 | 0, 0 |

## G. Paper Table II vs files
| Dataset | Quantity | Paper | File | Same? |
|---|---|---|---|---|
| MELD | Dialogues train+val / test | 1,153 / 280 | 1,152 / 280 | **train+val differs by 1** |
| MELD | Utterances train+val / test | 11,098 / 2,610 | 11,098 / 2,610 | yes |
| CMU-MOSEI | Dialogues train+val / test | 2,549 / 646 | 2,595 / 693 | **no (+46 / +47)** |
| CMU-MOSEI | Utterances train+val / test | 18,198 / 4,662 | 19,050 / 4,184 | **no (+852 / -478)** |

MELD: all utterances match, one dialogue fewer than the paper; cause unknown (not investigated). CMU-MOSEI: the README file
disagrees with Table II in every count. Total utterances are 23,234 in the file against 22,860 in the paper, and the train/test
balance differs. The Drive listing dates this file 18 Jun 2025, later than the MELD and IEMOCAP-6 files (19 Jan 2025), so it may be
a different or updated version than the one used for the paper; not verified.

## H. Class counts as the code sees them
MELD (index map from `dataloader.py`: 0 neutral, 1 surprise, 2 fear, 3 sadness, 4 joy, 5 disgust, 6 anger = paper Table IV column order):
| | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Train+val | 5,180 | 1,355 | 308 | 794 | 1,906 | 293 | 1,262 |
| Test | 1,256 | 281 | 50 | 208 | 402 | 68 | 345 |
MELD sentiment (0 neutral, 1 positive, 2 negative per `compare_to_paper.py`): train+val 5,180 / 2,567 / 3,351; test 1,256 / 521 / 833.
Sentiment class 0 equals emotion class 0 (neutral) in count, as expected.

CMU-MOSEI 7 classes (thresholds in `CMUMOSEIDataset7`, HNeg to HPos):
| | HNeg | Neg | WNeg | Neu | WPos | Pos | HPos |
|---|---|---|---|---|---|---|---|
| Train+val | 799 | 2,093 | 3,007 | 4,173 | 5,896 | 2,530 | 552 |
| Test | 22 | 159 | 602 | 921 | 1,666 | 692 | 122 |
CMU-MOSEI sentiment (0 negative, 1 neutral, 2 positive): train+val 5,899 / 4,173 / 8,978; test 783 / 921 / 2,480.
Highly Negative has only 22 test utterances, which fits the paper's 0.00 F1 on that class.

## Differences found, all datasets checked so far
| Item | Result |
|---|---|
| IEMOCAP-6 | server file = README file (md5 equal); matches Table II |
| IEMOCAP-4 | README file matches Table II and `run.py` dims; server copy not hashed |
| MELD | README file matches `run.py` dims and Table IV label order; Table II dialogues differ by 1 (1,152 vs 1,153); server copy not hashed |
| CMU-MOSEI | README file matches `run.py` dims but **not Table II counts** (dialogues and utterances both differ); server copy not hashed |

## To finish
Run on the server, then paste the output:
`grep -n "_path = " run.py` for the four dataset paths, then `md5sum` on each path. Equal md5 to the values above means the
experiments used the README files; if not, the files differ and the table needs a row per file.
