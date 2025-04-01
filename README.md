# tomfuns

This package contains some functions that may be helpful in generating stimuli, analyzing EEG data, and making plots.
Install with `pip install git+https://github.com/tomstoll/tomfuns.git`

Created by Thomas Stoll, primarily for use in the Maddox lab.

---
`tomfuns.analysis` contains functions I frequently use when analyzing EEG data, currently just helper functions for filtering. I plan to include functions to calculate responses through deconvolution or cross correlation in the future.

`tomfuns.plotting` contains functions to help make nice plots, with functions for setting defaults, getting some colors we commonly use, or getting the Unicode string for a Greek letter.

`tomfuns.stimuli` contains functions to generate stimuli, such as pip or click trains.

`tomfuns.utils` includes some useful miscellaneous functions, such as `checkpath` which checks if a path exists and creates it if it doesn't and `bin_4s8s_to_dec` which converts a binary number encoded with 4s and 8s to its decimal representation.

---
### To do:

- [ ] Add example scripts, demonstrating the use of each function.
- [ ] Define dependencies.
- [ ] Add functions to compute responses through deconvolution and cross correlation.
  - [ ] Include bayesian weighting option, with calculation of weights in a separate function.
- [ ] Add function to make/update a matplotlib stylesheet.
- [ ] Set the version to automatically increment (or use the git hash).