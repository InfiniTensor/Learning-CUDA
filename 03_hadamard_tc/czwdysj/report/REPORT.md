# Fused Hadamard + INT8 vs PyTorch

Mathematics: normalized Sylvester/Walsh Hadamard followed by rowwise dynamic symmetric INT8. CUDA-event latency uses three-run medians; allocation/input construction/correctness are outside timing. PyTorch eager includes Dao FHT and every standard quantization tensor operation; compiled compiles only quantization.

## FP32

| Shape | Implementation | us | Speed vs eager | Speed vs compiled | Speedup vs separate | Stable | Correct |
|---|---|---:|---:|---:|---:|---|---|
| [1,1024] | fused_v0 | 15.862 | 10.190x | 7.635x | 1.714x | UNSTABLE | PASS |
| [1,1024] | fused_v1 | 13.855 | 11.666x | 8.740x | 1.962x | UNSTABLE | PASS |
| [1,1024] | fused_v2 | 13.471 | 11.998x | 8.989x | 2.018x | UNSTABLE | PASS |
| [1,1024] | pytorch_compiled | 121.099 | 1.335x | 1.000x | 0.224x | UNSTABLE | PASS |
| [1,1024] | pytorch_eager | 161.634 | 1.000x | 0.749x | 0.168x | UNSTABLE | PASS |
| [1,1024] | separate | 27.186 | 5.945x | 4.454x | 1.000x | UNSTABLE | PASS |
| [1,1024] | torch_clone | 13.152 | 12.290x | 9.208x | 2.067x | yes | PASS |
| [32,128] | fused_v0 | 20.446 | 13.217x | 7.722x | 2.107x | UNSTABLE | PASS |
| [32,128] | fused_v1 | 20.037 | 13.487x | 7.880x | 2.150x | UNSTABLE | PASS |
| [32,128] | fused_v2 | 23.153 | 11.672x | 6.819x | 1.861x | UNSTABLE | PASS |
| [32,128] | pytorch_compiled | 157.888 | 1.712x | 1.000x | 0.273x | UNSTABLE | PASS |
| [32,128] | pytorch_eager | 270.243 | 1.000x | 0.584x | 0.159x | UNSTABLE | PASS |
| [32,128] | separate | 43.084 | 6.272x | 3.665x | 1.000x | UNSTABLE | PASS |
| [32,128] | torch_clone | 24.402 | 11.074x | 6.470x | 1.766x | UNSTABLE | PASS |
| [32,256] | fused_v0 | 14.794 | 13.946x | 8.330x | 2.697x | UNSTABLE | PASS |
| [32,256] | fused_v1 | 17.383 | 11.869x | 7.089x | 2.296x | UNSTABLE | PASS |
| [32,256] | fused_v2 | 17.075 | 12.083x | 7.217x | 2.337x | UNSTABLE | PASS |
| [32,256] | pytorch_compiled | 123.232 | 1.674x | 1.000x | 0.324x | UNSTABLE | PASS |
| [32,256] | pytorch_eager | 206.326 | 1.000x | 0.597x | 0.193x | yes | PASS |
| [32,256] | separate | 39.905 | 5.170x | 3.088x | 1.000x | UNSTABLE | PASS |
| [32,256] | torch_clone | 13.855 | 14.891x | 8.894x | 2.880x | UNSTABLE | PASS |
| [32,512] | fused_v0 | 13.257 | 13.485x | 9.066x | 2.375x | UNSTABLE | PASS |
| [32,512] | fused_v1 | 17.762 | 10.065x | 6.767x | 1.773x | UNSTABLE | PASS |
| [32,512] | fused_v2 | 13.594 | 13.151x | 8.842x | 2.317x | UNSTABLE | PASS |
| [32,512] | pytorch_compiled | 120.194 | 1.487x | 1.000x | 0.262x | UNSTABLE | PASS |
| [32,512] | pytorch_eager | 178.772 | 1.000x | 0.672x | 0.176x | yes | PASS |
| [32,512] | separate | 31.491 | 5.677x | 3.817x | 1.000x | UNSTABLE | PASS |
| [32,512] | torch_clone | 14.766 | 12.107x | 8.140x | 2.133x | UNSTABLE | PASS |
| [32,1024] | fused_v0 | 12.614 | 15.272x | 8.798x | 2.544x | UNSTABLE | PASS |
| [32,1024] | fused_v1 | 16.665 | 11.560x | 6.659x | 1.925x | UNSTABLE | PASS |
| [32,1024] | fused_v2 | 17.493 | 11.013x | 6.344x | 1.834x | UNSTABLE | PASS |
| [32,1024] | pytorch_compiled | 110.982 | 1.736x | 1.000x | 0.289x | UNSTABLE | PASS |
| [32,1024] | pytorch_eager | 192.647 | 1.000x | 0.576x | 0.167x | UNSTABLE | PASS |
| [32,1024] | separate | 32.088 | 6.004x | 3.459x | 1.000x | UNSTABLE | PASS |
| [32,1024] | torch_clone | 11.776 | 16.359x | 9.424x | 2.725x | UNSTABLE | PASS |
| [32,2048] | fused_v0 | 13.483 | 12.331x | 8.534x | 2.360x | UNSTABLE | PASS |
| [32,2048] | fused_v1 | 13.049 | 12.741x | 8.818x | 2.438x | yes | PASS |
| [32,2048] | fused_v2 | 13.381 | 12.425x | 8.599x | 2.378x | yes | PASS |
| [32,2048] | pytorch_compiled | 115.067 | 1.445x | 1.000x | 0.277x | yes | PASS |
| [32,2048] | pytorch_eager | 166.260 | 1.000x | 0.692x | 0.191x | UNSTABLE | PASS |
| [32,2048] | separate | 31.818 | 5.225x | 3.616x | 1.000x | UNSTABLE | PASS |
| [32,2048] | torch_clone | 12.743 | 13.047x | 9.030x | 2.497x | UNSTABLE | PASS |
| [32,4096] | fused_v0 | 14.946 | 11.648x | 7.852x | 2.139x | yes | PASS |
| [32,4096] | fused_v1 | 15.722 | 11.073x | 7.464x | 2.033x | UNSTABLE | PASS |
| [32,4096] | fused_v2 | 14.316 | 12.161x | 8.198x | 2.233x | UNSTABLE | PASS |
| [32,4096] | pytorch_compiled | 117.353 | 1.483x | 1.000x | 0.272x | UNSTABLE | PASS |
| [32,4096] | pytorch_eager | 174.092 | 1.000x | 0.674x | 0.184x | UNSTABLE | PASS |
| [32,4096] | separate | 31.967 | 5.446x | 3.671x | 1.000x | UNSTABLE | PASS |
| [32,4096] | torch_clone | 13.108 | 13.282x | 8.953x | 2.439x | UNSTABLE | PASS |
| [32,8192] | fused_v0 | 25.359 | 6.164x | 4.631x | 1.644x | UNSTABLE | PASS |
| [32,8192] | fused_v1 | 21.434 | 7.292x | 5.479x | 1.944x | UNSTABLE | PASS |
| [32,8192] | fused_v2 | 19.825 | 7.884x | 5.923x | 2.102x | yes | PASS |
| [32,8192] | pytorch_compiled | 117.432 | 1.331x | 1.000x | 0.355x | UNSTABLE | PASS |
| [32,8192] | pytorch_eager | 156.304 | 1.000x | 0.751x | 0.267x | UNSTABLE | PASS |
| [32,8192] | separate | 41.679 | 3.750x | 2.818x | 1.000x | yes | PASS |
| [32,8192] | torch_clone | 12.941 | 12.078x | 9.074x | 3.221x | UNSTABLE | PASS |
| [128,1024] | fused_v0 | 16.132 | 11.326x | 7.612x | 2.788x | UNSTABLE | PASS |
| [128,1024] | fused_v1 | 15.233 | 11.994x | 8.061x | 2.952x | UNSTABLE | PASS |
| [128,1024] | fused_v2 | 16.029 | 11.398x | 7.661x | 2.806x | yes | PASS |
| [128,1024] | pytorch_compiled | 122.802 | 1.488x | 1.000x | 0.366x | UNSTABLE | PASS |
| [128,1024] | pytorch_eager | 182.705 | 1.000x | 0.672x | 0.246x | UNSTABLE | PASS |
| [128,1024] | separate | 44.976 | 4.062x | 2.730x | 1.000x | UNSTABLE | PASS |
| [128,1024] | torch_clone | 13.125 | 13.920x | 9.356x | 3.427x | UNSTABLE | PASS |
| [1024,1024] | fused_v0 | 52.687 | 2.932x | 2.044x | 1.686x | yes | PASS |
| [1024,1024] | fused_v1 | 50.717 | 3.046x | 2.124x | 1.752x | yes | PASS |
| [1024,1024] | fused_v2 | 39.863 | 3.876x | 2.702x | 2.229x | yes | PASS |
| [1024,1024] | pytorch_compiled | 107.705 | 1.434x | 1.000x | 0.825x | UNSTABLE | PASS |
| [1024,1024] | pytorch_eager | 154.492 | 1.000x | 0.697x | 0.575x | UNSTABLE | PASS |
| [1024,1024] | separate | 88.849 | 1.739x | 1.212x | 1.000x | yes | PASS |
| [1024,1024] | torch_clone | 12.361 | 12.498x | 8.713x | 7.188x | UNSTABLE | PASS |
| [4096,1024] | fused_v0 | 219.112 | 1.951x | 0.627x | 1.767x | UNSTABLE | PASS |
| [4096,1024] | fused_v1 | 208.723 | 2.048x | 0.659x | 1.855x | UNSTABLE | PASS |
| [4096,1024] | fused_v2 | 167.607 | 2.551x | 0.820x | 2.310x | UNSTABLE | PASS |
| [4096,1024] | pytorch_compiled | 137.464 | 3.110x | 1.000x | 2.816x | UNSTABLE | PASS |
| [4096,1024] | pytorch_eager | 427.557 | 1.000x | 0.322x | 0.905x | UNSTABLE | PASS |
| [4096,1024] | separate | 387.135 | 1.104x | 0.355x | 1.000x | UNSTABLE | PASS |
| [4096,1024] | torch_clone | 38.016 | 11.247x | 3.616x | 10.184x | UNSTABLE | PASS |
| [4096,8192] | fused_v0 | 2463.766 | 2.235x | 0.603x | 1.295x | UNSTABLE | PASS |
| [4096,8192] | fused_v1 | 1751.358 | 3.145x | 0.849x | 1.822x | yes | PASS |
| [4096,8192] | fused_v2 | 1447.159 | 3.806x | 1.027x | 2.205x | UNSTABLE | PASS |
| [4096,8192] | pytorch_compiled | 1486.475 | 3.705x | 1.000x | 2.147x | yes | PASS |
| [4096,8192] | pytorch_eager | 5507.287 | 1.000x | 0.270x | 0.579x | yes | PASS |
| [4096,8192] | separate | 3191.062 | 1.726x | 0.466x | 1.000x | UNSTABLE | PASS |
| [4096,8192] | torch_clone | 858.321 | 6.416x | 1.732x | 3.718x | yes | PASS |

## FP16

| Shape | Implementation | us | Speed vs eager | Speed vs compiled | Speedup vs separate | Stable | Correct |
|---|---|---:|---:|---:|---:|---|---|
| [1,1024] | fused_v0 | 13.963 | 14.124x | 8.249x | 2.273x | UNSTABLE | PASS |
| [1,1024] | fused_v1 | 14.190 | 13.898x | 8.118x | 2.237x | UNSTABLE | PASS |
| [1,1024] | fused_v2 | 13.649 | 14.449x | 8.439x | 2.326x | UNSTABLE | PASS |
| [1,1024] | pytorch_compiled | 115.189 | 1.712x | 1.000x | 0.276x | UNSTABLE | PASS |
| [1,1024] | pytorch_eager | 197.212 | 1.000x | 0.584x | 0.161x | UNSTABLE | PASS |
| [1,1024] | separate | 31.744 | 6.213x | 3.629x | 1.000x | UNSTABLE | PASS |
| [1,1024] | torch_clone | 12.828 | 15.373x | 8.979x | 2.475x | yes | PASS |
| [32,128] | fused_v0 | 14.827 | 12.202x | 8.347x | 2.002x | UNSTABLE | PASS |
| [32,128] | fused_v1 | 13.347 | 13.555x | 9.272x | 2.224x | UNSTABLE | PASS |
| [32,128] | fused_v2 | 13.009 | 13.908x | 9.514x | 2.282x | UNSTABLE | PASS |
| [32,128] | pytorch_compiled | 123.760 | 1.462x | 1.000x | 0.240x | UNSTABLE | PASS |
| [32,128] | pytorch_eager | 180.919 | 1.000x | 0.684x | 0.164x | UNSTABLE | PASS |
| [32,128] | separate | 29.688 | 6.094x | 4.169x | 1.000x | UNSTABLE | PASS |
| [32,128] | torch_clone | 13.899 | 13.017x | 8.904x | 2.136x | UNSTABLE | PASS |
| [32,256] | fused_v0 | 14.377 | 13.708x | 8.312x | 2.259x | UNSTABLE | PASS |
| [32,256] | fused_v1 | 14.281 | 13.800x | 8.368x | 2.275x | yes | PASS |
| [32,256] | fused_v2 | 14.322 | 13.760x | 8.344x | 2.268x | yes | PASS |
| [32,256] | pytorch_compiled | 119.495 | 1.649x | 1.000x | 0.272x | UNSTABLE | PASS |
| [32,256] | pytorch_eager | 197.071 | 1.000x | 0.606x | 0.165x | yes | PASS |
| [32,256] | separate | 32.481 | 6.067x | 3.679x | 1.000x | UNSTABLE | PASS |
| [32,256] | torch_clone | 13.668 | 14.419x | 8.743x | 2.377x | UNSTABLE | PASS |
| [32,512] | fused_v0 | 13.806 | 14.256x | 8.588x | 2.473x | UNSTABLE | PASS |
| [32,512] | fused_v1 | 14.629 | 13.454x | 8.105x | 2.334x | UNSTABLE | PASS |
| [32,512] | fused_v2 | 13.861 | 14.199x | 8.554x | 2.463x | UNSTABLE | PASS |
| [32,512] | pytorch_compiled | 118.565 | 1.660x | 1.000x | 0.288x | UNSTABLE | PASS |
| [32,512] | pytorch_eager | 196.819 | 1.000x | 0.602x | 0.173x | yes | PASS |
| [32,512] | separate | 34.143 | 5.765x | 3.473x | 1.000x | UNSTABLE | PASS |
| [32,512] | torch_clone | 14.062 | 13.996x | 8.431x | 2.428x | UNSTABLE | PASS |
| [32,1024] | fused_v0 | 14.306 | 13.481x | 8.407x | 2.254x | UNSTABLE | PASS |
| [32,1024] | fused_v1 | 15.355 | 12.560x | 7.833x | 2.100x | UNSTABLE | PASS |
| [32,1024] | fused_v2 | 14.011 | 13.765x | 8.584x | 2.301x | UNSTABLE | PASS |
| [32,1024] | pytorch_compiled | 120.273 | 1.604x | 1.000x | 0.268x | yes | PASS |
| [32,1024] | pytorch_eager | 192.859 | 1.000x | 0.624x | 0.167x | UNSTABLE | PASS |
| [32,1024] | separate | 32.246 | 5.981x | 3.730x | 1.000x | UNSTABLE | PASS |
| [32,1024] | torch_clone | 12.931 | 14.915x | 9.301x | 2.494x | UNSTABLE | PASS |
| [32,2048] | fused_v0 | 15.550 | 12.362x | 7.395x | 1.841x | UNSTABLE | PASS |
| [32,2048] | fused_v1 | 14.200 | 13.538x | 8.099x | 2.016x | UNSTABLE | PASS |
| [32,2048] | fused_v2 | 15.710 | 12.236x | 7.320x | 1.822x | UNSTABLE | PASS |
| [32,2048] | pytorch_compiled | 114.997 | 1.672x | 1.000x | 0.249x | UNSTABLE | PASS |
| [32,2048] | pytorch_eager | 192.230 | 1.000x | 0.598x | 0.149x | UNSTABLE | PASS |
| [32,2048] | separate | 28.621 | 6.716x | 4.018x | 1.000x | yes | PASS |
| [32,2048] | torch_clone | 13.587 | 14.148x | 8.464x | 2.106x | UNSTABLE | PASS |
| [32,4096] | fused_v0 | 17.834 | 10.984x | 6.598x | 2.018x | UNSTABLE | PASS |
| [32,4096] | fused_v1 | 14.803 | 13.233x | 7.949x | 2.431x | UNSTABLE | PASS |
| [32,4096] | fused_v2 | 15.058 | 13.008x | 7.814x | 2.389x | UNSTABLE | PASS |
| [32,4096] | pytorch_compiled | 117.666 | 1.665x | 1.000x | 0.306x | UNSTABLE | PASS |
| [32,4096] | pytorch_eager | 195.887 | 1.000x | 0.601x | 0.184x | yes | PASS |
| [32,4096] | separate | 35.982 | 5.444x | 3.270x | 1.000x | UNSTABLE | PASS |
| [32,4096] | torch_clone | 13.172 | 14.871x | 8.933x | 2.732x | yes | PASS |
| [32,8192] | fused_v0 | 26.985 | 6.718x | 4.458x | 1.652x | yes | PASS |
| [32,8192] | fused_v1 | 21.627 | 8.382x | 5.562x | 2.062x | UNSTABLE | PASS |
| [32,8192] | fused_v2 | 20.493 | 8.846x | 5.870x | 2.176x | yes | PASS |
| [32,8192] | pytorch_compiled | 120.297 | 1.507x | 1.000x | 0.371x | UNSTABLE | PASS |
| [32,8192] | pytorch_eager | 181.282 | 1.000x | 0.664x | 0.246x | UNSTABLE | PASS |
| [32,8192] | separate | 44.590 | 4.065x | 2.698x | 1.000x | UNSTABLE | PASS |
| [32,8192] | torch_clone | 12.694 | 14.281x | 9.477x | 3.513x | UNSTABLE | PASS |
| [128,1024] | fused_v0 | 14.882 | 13.227x | 8.064x | 2.162x | UNSTABLE | PASS |
| [128,1024] | fused_v1 | 14.258 | 13.806x | 8.416x | 2.256x | UNSTABLE | PASS |
| [128,1024] | fused_v2 | 13.917 | 14.144x | 8.623x | 2.312x | UNSTABLE | PASS |
| [128,1024] | pytorch_compiled | 120.003 | 1.640x | 1.000x | 0.268x | yes | PASS |
| [128,1024] | pytorch_eager | 196.843 | 1.000x | 0.610x | 0.163x | yes | PASS |
| [128,1024] | separate | 32.169 | 6.119x | 3.730x | 1.000x | UNSTABLE | PASS |
| [128,1024] | torch_clone | 13.221 | 14.889x | 9.077x | 2.433x | UNSTABLE | PASS |
| [1024,1024] | fused_v0 | 56.136 | 3.253x | 2.164x | 1.694x | yes | PASS |
| [1024,1024] | fused_v1 | 57.730 | 3.163x | 2.104x | 1.647x | UNSTABLE | PASS |
| [1024,1024] | fused_v2 | 44.701 | 4.085x | 2.717x | 2.128x | UNSTABLE | PASS |
| [1024,1024] | pytorch_compiled | 121.457 | 1.503x | 1.000x | 0.783x | yes | PASS |
| [1024,1024] | pytorch_eager | 182.589 | 1.000x | 0.665x | 0.521x | UNSTABLE | PASS |
| [1024,1024] | separate | 95.108 | 1.920x | 1.277x | 1.000x | UNSTABLE | PASS |
| [1024,1024] | torch_clone | 13.306 | 13.723x | 9.128x | 7.148x | UNSTABLE | PASS |
| [4096,1024] | fused_v0 | 243.151 | 1.854x | 0.487x | 1.693x | UNSTABLE | PASS |
| [4096,1024] | fused_v1 | 227.263 | 1.984x | 0.521x | 1.811x | yes | PASS |
| [4096,1024] | fused_v2 | 180.523 | 2.497x | 0.656x | 2.280x | UNSTABLE | PASS |
| [4096,1024] | pytorch_compiled | 118.345 | 3.809x | 1.000x | 3.478x | UNSTABLE | PASS |
| [4096,1024] | pytorch_eager | 450.789 | 1.000x | 0.263x | 0.913x | UNSTABLE | PASS |
| [4096,1024] | separate | 411.590 | 1.095x | 0.288x | 1.000x | UNSTABLE | PASS |
| [4096,1024] | torch_clone | 16.224 | 27.785x | 7.294x | 25.369x | UNSTABLE | PASS |
| [4096,8192] | fused_v0 | 2534.323 | 2.389x | 0.396x | 1.308x | UNSTABLE | PASS |
| [4096,8192] | fused_v1 | 1775.723 | 3.410x | 0.565x | 1.867x | yes | PASS |
| [4096,8192] | fused_v2 | 1473.238 | 4.110x | 0.681x | 2.250x | UNSTABLE | PASS |
| [4096,8192] | pytorch_compiled | 1003.035 | 6.036x | 1.000x | 3.305x | UNSTABLE | PASS |
| [4096,8192] | pytorch_eager | 6054.712 | 1.000x | 0.166x | 0.547x | yes | PASS |
| [4096,8192] | separate | 3314.591 | 1.827x | 0.303x | 1.000x | UNSTABLE | PASS |
| [4096,8192] | torch_clone | 521.500 | 11.610x | 1.923x | 6.356x | UNSTABLE | PASS |

## BF16

| Shape | Implementation | us | Speed vs eager | Speed vs compiled | Speedup vs separate | Stable | Correct |
|---|---|---:|---:|---:|---:|---|---|
| [1,1024] | fused_v0 | 13.297 | 14.196x | 8.274x | 2.670x | yes | PASS |
| [1,1024] | fused_v1 | 13.963 | 13.519x | 7.879x | 2.543x | UNSTABLE | PASS |
| [1,1024] | fused_v2 | 14.264 | 13.234x | 7.713x | 2.489x | yes | PASS |
| [1,1024] | pytorch_compiled | 110.019 | 1.716x | 1.000x | 0.323x | UNSTABLE | PASS |
| [1,1024] | pytorch_eager | 188.764 | 1.000x | 0.583x | 0.188x | UNSTABLE | PASS |
| [1,1024] | separate | 35.507 | 5.316x | 3.099x | 1.000x | UNSTABLE | PASS |
| [1,1024] | torch_clone | 12.390 | 15.235x | 8.879x | 2.866x | UNSTABLE | PASS |
| [32,128] | fused_v0 | 14.246 | 11.896x | 7.699x | 2.092x | UNSTABLE | PASS |
| [32,128] | fused_v1 | 12.211 | 13.879x | 8.982x | 2.441x | UNSTABLE | PASS |
| [32,128] | fused_v2 | 11.871 | 14.276x | 9.239x | 2.511x | UNSTABLE | PASS |
| [32,128] | pytorch_compiled | 109.679 | 1.545x | 1.000x | 0.272x | UNSTABLE | PASS |
| [32,128] | pytorch_eager | 169.473 | 1.000x | 0.647x | 0.176x | UNSTABLE | PASS |
| [32,128] | separate | 29.804 | 5.686x | 3.680x | 1.000x | UNSTABLE | PASS |
| [32,128] | torch_clone | 12.757 | 13.285x | 8.598x | 2.336x | UNSTABLE | PASS |
| [32,256] | fused_v0 | 13.963 | 13.988x | 8.142x | 2.207x | UNSTABLE | PASS |
| [32,256] | fused_v1 | 13.810 | 14.143x | 8.233x | 2.232x | UNSTABLE | PASS |
| [32,256] | fused_v2 | 13.553 | 14.411x | 8.388x | 2.274x | UNSTABLE | PASS |
| [32,256] | pytorch_compiled | 113.690 | 1.718x | 1.000x | 0.271x | yes | PASS |
| [32,256] | pytorch_eager | 195.314 | 1.000x | 0.582x | 0.158x | yes | PASS |
| [32,256] | separate | 30.818 | 6.338x | 3.689x | 1.000x | yes | PASS |
| [32,256] | torch_clone | 12.656 | 15.433x | 8.983x | 2.435x | UNSTABLE | PASS |
| [32,512] | fused_v0 | 13.626 | 14.625x | 8.627x | 2.358x | UNSTABLE | PASS |
| [32,512] | fused_v1 | 13.505 | 14.756x | 8.705x | 2.379x | UNSTABLE | PASS |
| [32,512] | fused_v2 | 14.361 | 13.876x | 8.186x | 2.237x | UNSTABLE | PASS |
| [32,512] | pytorch_compiled | 117.556 | 1.695x | 1.000x | 0.273x | UNSTABLE | PASS |
| [32,512] | pytorch_eager | 199.272 | 1.000x | 0.590x | 0.161x | yes | PASS |
| [32,512] | separate | 32.127 | 6.203x | 3.659x | 1.000x | UNSTABLE | PASS |
| [32,512] | torch_clone | 13.210 | 15.085x | 8.899x | 2.432x | UNSTABLE | PASS |
| [32,1024] | fused_v0 | 13.775 | 13.705x | 8.639x | 2.302x | UNSTABLE | PASS |
| [32,1024] | fused_v1 | 13.497 | 13.986x | 8.817x | 2.350x | yes | PASS |
| [32,1024] | fused_v2 | 13.657 | 13.823x | 8.714x | 2.322x | yes | PASS |
| [32,1024] | pytorch_compiled | 119.003 | 1.586x | 1.000x | 0.266x | yes | PASS |
| [32,1024] | pytorch_eager | 188.779 | 1.000x | 0.630x | 0.168x | yes | PASS |
| [32,1024] | separate | 31.712 | 5.953x | 3.753x | 1.000x | UNSTABLE | PASS |
| [32,1024] | torch_clone | 12.671 | 14.898x | 9.392x | 2.503x | UNSTABLE | PASS |
| [32,2048] | fused_v0 | 15.593 | 12.551x | 8.064x | 2.083x | UNSTABLE | PASS |
| [32,2048] | fused_v1 | 14.247 | 13.737x | 8.825x | 2.280x | UNSTABLE | PASS |
| [32,2048] | fused_v2 | 14.229 | 13.755x | 8.837x | 2.283x | UNSTABLE | PASS |
| [32,2048] | pytorch_compiled | 125.739 | 1.557x | 1.000x | 0.258x | yes | PASS |
| [32,2048] | pytorch_eager | 195.712 | 1.000x | 0.642x | 0.166x | UNSTABLE | PASS |
| [32,2048] | separate | 32.477 | 6.026x | 3.872x | 1.000x | UNSTABLE | PASS |
| [32,2048] | torch_clone | 15.748 | 12.428x | 7.984x | 2.062x | UNSTABLE | PASS |
| [32,4096] | fused_v0 | 16.711 | 11.248x | 6.795x | 2.183x | UNSTABLE | PASS |
| [32,4096] | fused_v1 | 15.189 | 12.375x | 7.476x | 2.402x | UNSTABLE | PASS |
| [32,4096] | fused_v2 | 14.413 | 13.042x | 7.878x | 2.532x | UNSTABLE | PASS |
| [32,4096] | pytorch_compiled | 113.547 | 1.655x | 1.000x | 0.321x | UNSTABLE | PASS |
| [32,4096] | pytorch_eager | 187.967 | 1.000x | 0.604x | 0.194x | yes | PASS |
| [32,4096] | separate | 36.487 | 5.152x | 3.112x | 1.000x | UNSTABLE | PASS |
| [32,4096] | torch_clone | 13.079 | 14.371x | 8.681x | 2.790x | UNSTABLE | PASS |
| [32,8192] | fused_v0 | 26.803 | 6.910x | 4.157x | 1.659x | yes | PASS |
| [32,8192] | fused_v1 | 22.667 | 8.171x | 4.915x | 1.962x | yes | PASS |
| [32,8192] | fused_v2 | 20.452 | 9.056x | 5.447x | 2.174x | yes | PASS |
| [32,8192] | pytorch_compiled | 111.408 | 1.663x | 1.000x | 0.399x | yes | PASS |
| [32,8192] | pytorch_eager | 185.219 | 1.000x | 0.601x | 0.240x | UNSTABLE | PASS |
| [32,8192] | separate | 44.468 | 4.165x | 2.505x | 1.000x | UNSTABLE | PASS |
| [32,8192] | torch_clone | 12.277 | 15.087x | 9.075x | 3.622x | UNSTABLE | PASS |
| [128,1024] | fused_v0 | 13.768 | 13.594x | 8.289x | 2.753x | UNSTABLE | PASS |
| [128,1024] | fused_v1 | 15.729 | 11.899x | 7.256x | 2.410x | UNSTABLE | PASS |
| [128,1024] | fused_v2 | 15.006 | 12.472x | 7.605x | 2.526x | UNSTABLE | PASS |
| [128,1024] | pytorch_compiled | 114.130 | 1.640x | 1.000x | 0.332x | yes | PASS |
| [128,1024] | pytorch_eager | 187.164 | 1.000x | 0.610x | 0.203x | yes | PASS |
| [128,1024] | separate | 37.909 | 4.937x | 3.011x | 1.000x | UNSTABLE | PASS |
| [128,1024] | torch_clone | 12.556 | 14.906x | 9.090x | 3.019x | yes | PASS |
| [1024,1024] | fused_v0 | 56.657 | 3.360x | 2.051x | 1.778x | UNSTABLE | PASS |
| [1024,1024] | fused_v1 | 55.723 | 3.416x | 2.085x | 1.808x | UNSTABLE | PASS |
| [1024,1024] | fused_v2 | 44.755 | 4.253x | 2.596x | 2.251x | yes | PASS |
| [1024,1024] | pytorch_compiled | 116.184 | 1.638x | 1.000x | 0.867x | UNSTABLE | PASS |
| [1024,1024] | pytorch_eager | 190.353 | 1.000x | 0.610x | 0.529x | UNSTABLE | PASS |
| [1024,1024] | separate | 100.738 | 1.890x | 1.153x | 1.000x | UNSTABLE | PASS |
| [1024,1024] | torch_clone | 14.308 | 13.304x | 8.120x | 7.041x | UNSTABLE | PASS |
| [4096,1024] | fused_v0 | 237.382 | 1.983x | 0.451x | 1.668x | yes | PASS |
| [4096,1024] | fused_v1 | 222.806 | 2.112x | 0.481x | 1.777x | yes | PASS |
| [4096,1024] | fused_v2 | 176.998 | 2.659x | 0.605x | 2.237x | UNSTABLE | PASS |
| [4096,1024] | pytorch_compiled | 107.126 | 4.393x | 1.000x | 3.696x | yes | PASS |
| [4096,1024] | pytorch_eager | 470.635 | 1.000x | 0.228x | 0.841x | yes | PASS |
| [4096,1024] | separate | 395.986 | 1.189x | 0.271x | 1.000x | yes | PASS |
| [4096,1024] | torch_clone | 16.301 | 28.872x | 6.572x | 24.292x | UNSTABLE | PASS |
| [4096,8192] | fused_v0 | 2466.868 | 2.438x | 0.418x | 1.334x | yes | PASS |
| [4096,8192] | fused_v1 | 1760.543 | 3.417x | 0.586x | 1.869x | yes | PASS |
| [4096,8192] | fused_v2 | 1454.339 | 4.136x | 0.710x | 2.262x | yes | PASS |
| [4096,8192] | pytorch_compiled | 1032.034 | 5.828x | 1.000x | 3.188x | UNSTABLE | PASS |
| [4096,8192] | pytorch_eager | 6014.929 | 1.000x | 0.172x | 0.547x | UNSTABLE | PASS |
| [4096,8192] | separate | 3290.441 | 1.828x | 0.314x | 1.000x | UNSTABLE | PASS |
| [4096,8192] | torch_clone | 500.913 | 12.008x | 2.060x | 6.569x | UNSTABLE | PASS |

## Aggregate

- fused_v0 vs PyTorch eager: geometric mean 8.046x, wins 36/36.
- fused_v0 vs PyTorch compiled: geometric mean 4.295x, wins 30/36.
- fused_v1 vs PyTorch eager: geometric mean 8.446x, wins 36/36.
- fused_v1 vs PyTorch compiled: geometric mean 4.508x, wins 30/36.
- fused_v2 vs PyTorch eager: geometric mean 9.054x, wins 36/36.
- fused_v2 vs PyTorch compiled: geometric mean 4.833x, wins 31/36.

## Numerical note

Dao returns FP16/BF16 Hadamard output in the input dtype before FP32 quantization, while fused kernels keep butterfly intermediates in FP32. Ratios are valid only when final INT8, amax, and scale pass the recorded gate.

## NCU pipeline analysis

Profile workload: FP16 `[1024,4096]`, after 100 warmup iterations, one complete pipeline operation. Full NCU reports use `--set full`; NCU replay time is diagnostic and is not used as benchmark latency.

| Pipeline | Kernels | Sum of per-kernel NCU duration | Main observation |
|---|---:|---:|---|
| PyTorch eager | 12 | 733.46 us | Dao FHT plus separate cast, abs, reduction, scale, divide, round, clamp and INT8 conversion kernels |
| PyTorch compiled | 2 | 155.23 us | Dao FHT plus one Triton fused reduction/quantization kernel |

- Eager launches 12 kernels. Its large elementwise/reduction stages reach roughly 95.0% DRAM throughput, showing repeated global-memory materialization rather than Hadamard arithmetic is the dominant avoidable cost.
- `torch.compile` reduces the quantization chain to `triton_red_fused__to_copy_abs_amax_clamp_div_gt_ones_like_round_unsqueeze_where_0`. The compiled quantization kernel takes 74.21 us in NCU, with 59.30% DRAM and 64.47% SM throughput.
- Dao FHT itself is essentially unchanged between runs (81.76 us eager and 81.02 us compiled). Compilation optimizes only the quantization stage, as intended by the fairness contract.
- The fused CUDA versions perform Hadamard, amax and INT8 conversion in one launch and avoid writing/reading the full Hadamard intermediate. Earlier full NCU reports for the same workload measured v0/v1/v2 at 295.10/264.67/216.19 us under NCU instrumentation; v2 remains limited by cross-warp shared-memory stages and SM work rather than DRAM bandwidth.
- Next optimization direction: preserve v2's single-launch dataflow while reducing cross-warp barriers, specialize small dimensions to process several rows per block, and test a register/shared-memory tiling strategy for D=4096–8192.

The reports `pytorch_eager.ncu-rep` and `pytorch_compiled.ncu-rep` were successfully imported by Windows Nsight Compute 2026.2 and opened in NCU-UI.
