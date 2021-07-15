# History of the models

- v1 : First generated model {Trash results: One chord}
> w LSTM.256 X2 + Dense.128 
 
 - v2: Smaller model. Smaller data set. Created new file {Better result, but still bad}
 > w LSTM.64 + Dense.128

- v3: Bigger model. Moved datafunctions to functions.py {No big improvement}
> w LSTM.64 X2 + Dense.128

- v4: Unknown change. Unknown model architecture

- v5: Trial of larger NN backend. Changed dropout thresholds. Ran 500ep on DL. Switched to Goldberg variations ONLY. {Goodish result}
>w LSTM.256 + DENSE.128 + DENSE.64

- v6: Everything larger. Decreased dropout threshold. Ran 100ep on DL. {Good results once Quantized by logic. Good progress. (L~3)}
>w LSTM.256 + LSTM.128 + DENSE.128 + DENSE.64

- v7: Changed file structure. GRU arch. Beat_resolution=8, lookback=*6. {Absolute Garbage (L~15)}
>w GRU.256 + DENSE.128 + DENSE.64

- v8: Same settings as v6 but GRU arch. {Incredible L~2.7e-7, but 5% accu. Generates nothing(?)}
Generation also went really funky
>w GRU.256 + GRU.64 + DENSE.128 + DENSE.64

- v9: Change loss function to mse from categorical_crossentropy {Low loss but acc of only 5%. Generated nothing}
>w GRU.256 + GRU.64 + DENSE.128 + DENSE.64

- v10: Back to LSTM but much larger {Loss of 2.8, acc of 47%}
>w LSTM.512 + LSTM.128 + DENSE.256 + DENSE.128

- v11: Make to v6, but beat_resolution=12, and train on "bwv653" An Wasserflussen Babylon - BWV 653
(By the Waters of Babylon) {Not bad results}
>w LSTM.256 + LSTM.128 + DENSE.128 + DENSE.64

- v12: Same as v11 but training on "bwv588" Canzona in D-Minor {Not bad results}
>w LSTM.256 + LSTM.128 + DENSE.128 + DENSE.64
 
- v13: Restarted after long hiatus. Rewrote most (if not all) functions and trained on deeplearn for 1 hour. Trained on Goldberg variations. 
Beat resolution: 24, Lookback: 4 bars. Okayish result
>w LSTM.128 + DENSE.128

- v14: Different loss function, used "binary_crossentropy". Trained on Goldberg on deeplearn
Beat resolution: 4, Lookback: 8 bars (Very good results... almost too good, might have memorised them)
>w LSTM.256 + LSTM.512 + LSTM.256 + DENSE.256 + DROPOUT-0.3

- v15: Use of time distributed layers to add a convnet ontop of the LSTM to get some higher level feature extraction.

>w [] --- FILL IN ----

- v?: Longer lookback (From 2 bars to 4 bars)
- v?: Back to v10 but smaller training data "bwv653"
- v?: Train on more data for longer.
- v?: Train on jazz
- v?: Use a convnet to try and grab some patterns to start
