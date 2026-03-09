# ECG file format for replay mode

Use a plain CSV or TXT file containing ECG samples as numbers.

Accepted examples:
- one sample per line
- comma-separated samples on one line
- mixed commas/newlines

Ignored values:
- non-numeric tokens (headers/text)

Example:
```
0.01
0.02
0.03
```
or
```
0.01,0.02,0.03
```
