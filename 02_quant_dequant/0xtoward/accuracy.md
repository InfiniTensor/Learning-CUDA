# Reconstruction errors

Metrics use FP32 reconstructed values and FP64 accumulation. Relative L2 = sqrt(sum((y-x)^2)/sum(x^2)).

| Tensor/distribution | Mode | Max abs | MAE | MSE | Relative L2 |
|---|---|---:|---:|---:|---:|
| uniform[-1,1] | mxfp8_floor | 0.1249993 | 0.01628034 | 0.0007944736 | 0.04880504 |
| uniform[-1,1] | mxfp8_rceil | 0.03125 | 0.01044065 | 0.0001866729 | 0.02365732 |
| uniform[-1,1] | nvfp4 | 0.1666651 | 0.04430572 | 0.003447313 | 0.1016636 |
| uniform[-1,1] | nvfp4_static6_bound256 | 0.1666651 | 0.04412164 | 0.003431613 | 0.1014318 |
| uniform[-1,1] | nvfp4_4over6 | 0.1666507 | 0.04016963 | 0.002538514 | 0.08723981 |
| normal(0,1) | mxfp8_floor | 0.497474 | 0.01868055 | 0.0008607751 | 0.029334 |
| normal(0,1) | mxfp8_rceil | 0.2453198 | 0.01795219 | 0.0007034151 | 0.02651748 |
| normal(0,1) | nvfp4 | 0.5745952 | 0.07144959 | 0.009049535 | 0.09511293 |
| normal(0,1) | nvfp4_static6_bound256 | 0.5976553 | 0.0714157 | 0.009051516 | 0.09512334 |
| normal(0,1) | nvfp4_4over6 | 0.5071628 | 0.06862351 | 0.007563903 | 0.08695597 |
| normal_with_1pct_30x_outliers | mxfp8_floor | 9.422203 | 0.0259986 | 0.01627265 | 0.03956886 |
| normal_with_1pct_30x_outliers | mxfp8_rceil | 4.800797 | 0.02322705 | 0.007156993 | 0.02624157 |
| normal_with_1pct_30x_outliers | nvfp4 | 4.735085 | 0.1314544 | 0.06811085 | 0.08095294 |
| normal_with_1pct_30x_outliers | nvfp4_static6_bound256 | 4.148308 | 0.1315421 | 0.06804847 | 0.08091586 |
| normal_with_1pct_30x_outliers | nvfp4_4over6 | 4.120773 | 0.1284329 | 0.06493614 | 0.07904378 |
| model.language_model.layers.0.linear_attn.in_proj_qkv.weight | mxfp8_floor | 0.02929688 | 0.000289481 | 2.508778e-07 | 0.03025517 |
| model.language_model.layers.0.linear_attn.in_proj_qkv.weight | mxfp8_rceil | 0.015625 | 0.0002764601 | 1.940509e-07 | 0.02660887 |
| model.language_model.layers.0.linear_attn.in_proj_qkv.weight | nvfp4 | 0.04348028 | 0.001105582 | 2.449822e-06 | 0.09454439 |
| model.language_model.layers.0.linear_attn.in_proj_qkv.weight | nvfp4_static6_bound256 | 0.04687503 | 0.001105911 | 2.451042e-06 | 0.09456794 |
| model.language_model.layers.0.linear_attn.in_proj_qkv.weight | nvfp4_4over6 | 0.04687503 | 0.001064115 | 2.066476e-06 | 0.08683279 |
| model.language_model.layers.0.linear_attn.in_proj_z.weight | mxfp8_floor | 0.02441406 | 0.0003088293 | 2.696406e-07 | 0.03001737 |
| model.language_model.layers.0.linear_attn.in_proj_z.weight | mxfp8_rceil | 0.0078125 | 0.0002962406 | 2.120473e-07 | 0.02661928 |
| model.language_model.layers.0.linear_attn.in_proj_z.weight | nvfp4 | 0.02599517 | 0.001183614 | 2.683073e-06 | 0.09468827 |
| model.language_model.layers.0.linear_attn.in_proj_z.weight | nvfp4_static6_bound256 | 0.02270509 | 0.00118428 | 2.683425e-06 | 0.09469449 |
| model.language_model.layers.0.linear_attn.in_proj_z.weight | nvfp4_4over6 | 0.02270509 | 0.001139728 | 2.262043e-06 | 0.08694214 |
| model.language_model.layers.0.linear_attn.out_proj.weight | mxfp8_floor | 0.02832031 | 0.0002225381 | 1.432724e-07 | 0.03059639 |
| model.language_model.layers.0.linear_attn.out_proj.weight | mxfp8_rceil | 0.015625 | 0.0002118332 | 1.082799e-07 | 0.02659885 |
| model.language_model.layers.0.linear_attn.out_proj.weight | nvfp4 | 0.02859934 | 0.0008537015 | 1.375468e-06 | 0.09480127 |
| model.language_model.layers.0.linear_attn.out_proj.weight | nvfp4_static6_bound256 | 0.02937827 | 0.000852584 | 1.372574e-06 | 0.09470148 |
| model.language_model.layers.0.linear_attn.out_proj.weight | nvfp4_4over6 | 0.02522787 | 0.000821965 | 1.158037e-06 | 0.08698615 |
| model.language_model.layers.0.mlp.down_proj.weight | mxfp8_floor | 0.01513672 | 0.0001546936 | 6.078608e-08 | 0.02901632 |
| model.language_model.layers.0.mlp.down_proj.weight | mxfp8_rceil | 0.01367188 | 0.0001501647 | 5.123143e-08 | 0.02663842 |
| model.language_model.layers.0.mlp.down_proj.weight | nvfp4 | 0.02394903 | 0.0005988051 | 6.508079e-07 | 0.0949438 |
| model.language_model.layers.0.mlp.down_proj.weight | nvfp4_static6_bound256 | 0.01920573 | 0.0005993759 | 6.515233e-07 | 0.09499597 |
| model.language_model.layers.0.mlp.down_proj.weight | nvfp4_4over6 | 0.01920573 | 0.0005760162 | 5.461526e-07 | 0.08697556 |
| model.language_model.layers.2.linear_attn.in_proj_qkv.weight | mxfp8_floor | 0.02246094 | 0.000307007 | 2.681233e-07 | 0.03000839 |
| model.language_model.layers.2.linear_attn.in_proj_qkv.weight | mxfp8_rceil | 0.0078125 | 0.0002938369 | 2.109994e-07 | 0.02662049 |
| model.language_model.layers.2.linear_attn.in_proj_qkv.weight | nvfp4 | 0.0266462 | 0.001171309 | 2.680901e-06 | 0.09488899 |
| model.language_model.layers.2.linear_attn.in_proj_qkv.weight | nvfp4_static6_bound256 | 0.02734375 | 0.001171992 | 2.682961e-06 | 0.09492544 |
| model.language_model.layers.2.linear_attn.in_proj_qkv.weight | nvfp4_4over6 | 0.02099609 | 0.001127062 | 2.249917e-06 | 0.08692779 |
| model.language_model.layers.2.linear_attn.in_proj_z.weight | mxfp8_floor | 0.01513672 | 0.0003458165 | 3.441371e-07 | 0.03006791 |
| model.language_model.layers.2.linear_attn.in_proj_z.weight | mxfp8_rceil | 0.0078125 | 0.0003311125 | 2.689575e-07 | 0.02658149 |
| model.language_model.layers.2.linear_attn.in_proj_z.weight | nvfp4 | 0.02267021 | 0.001321103 | 3.437087e-06 | 0.09502388 |
| model.language_model.layers.2.linear_attn.in_proj_z.weight | nvfp4_static6_bound256 | 0.02001953 | 0.001321106 | 3.44038e-06 | 0.09506939 |
| model.language_model.layers.2.linear_attn.in_proj_z.weight | nvfp4_4over6 | 0.01965332 | 0.001270136 | 2.876953e-06 | 0.08693684 |
| model.language_model.layers.9.mlp.gate_proj.weight | mxfp8_floor | 0.0078125 | 0.0001901595 | 1.025812e-07 | 0.03006246 |
| model.language_model.layers.9.mlp.gate_proj.weight | mxfp8_rceil | 0.0078125 | 0.000182464 | 8.028034e-08 | 0.02659471 |
| model.language_model.layers.9.mlp.gate_proj.weight | nvfp4 | 0.01311384 | 0.0007300836 | 1.019032e-06 | 0.09475116 |
| model.language_model.layers.9.mlp.gate_proj.weight | nvfp4_static6_bound256 | 0.01546223 | 0.0007298289 | 1.019018e-06 | 0.0947505 |
| model.language_model.layers.9.mlp.gate_proj.weight | nvfp4_4over6 | 0.01367188 | 0.0007030132 | 8.568161e-07 | 0.08688293 |
| model.language_model.layers.9.mlp.up_proj.weight | mxfp8_floor | 0.006835938 | 0.0001529496 | 6.193731e-08 | 0.0295304 |
| model.language_model.layers.9.mlp.up_proj.weight | mxfp8_rceil | 0.003417969 | 0.0001476071 | 5.039246e-08 | 0.02663642 |
| model.language_model.layers.9.mlp.up_proj.weight | nvfp4 | 0.008172899 | 0.0005908472 | 6.370429e-07 | 0.094706 |
| model.language_model.layers.9.mlp.up_proj.weight | nvfp4_static6_bound256 | 0.009745281 | 0.000590757 | 6.368287e-07 | 0.09469008 |
| model.language_model.layers.9.mlp.up_proj.weight | nvfp4_4over6 | 0.008026123 | 0.000569289 | 5.359629e-07 | 0.08686814 |
