# sentclass #

Currently a simple wrapper around a transformer model for subjectivity classification (`cffl/bert-base-styleclassification-subjective-neutral`), and a simple classifier based on concreteness word norms (Brysbaert et al.).

While my research isn't directly about subjectivity or concreteness, sometimes it's useful to filter sentences based on these dimensions.

## Install ##

`pip install git+https://github.com/mwestera/sentclass`

This will make the command `sentclass` available in your shell.

## Examples ##

```bash
$ echo "This will compute only the concreteness of this sentence" | sentclass --conc
$ echo "This will compute all attributes of this sentence." | sentclass
$ echo "This will compute only the subjectivity of this sentence" | sentclass --subj
```

Or pass a file into it:

```bash
$ sentclass sentences.txt --subj
```
