| sequence_length | model | domains | nmje_mean | nmje_sd | seeds | pck_at_0_05_mean | pck_at_0_05_sd | pck_at_0_10_mean | pck_at_0_10_sd | pck_at_0_20_mean | pck_at_0_20_sd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | geometric baseline | Indoor+Outdoor+Drone | 0.1586 | 0.0055 | 5 | 0.1527 | 0.0154 | 0.4174 | 0.0151 | 0.7794 | 0.0125 |
| 1 | mlp | Indoor+Outdoor+Drone | 0.0779 | 0.0041 | 5 | 0.4057 | 0.0275 | 0.7468 | 0.0212 | 0.9464 | 0.0128 |
| 1 | graph | Indoor+Outdoor+Drone | 0.0881 | 0.0033 | 5 | 0.3146 | 0.0145 | 0.6845 | 0.0177 | 0.9463 | 0.0041 |
| 16 | geometric baseline | Indoor+Outdoor | 0.1517 | 0.0045 | 5 | 0.1479 | 0.0116 | 0.4289 | 0.0238 | 0.7582 | 0.0170 |
| 16 | temporal interpolation | Indoor+Outdoor | 0.0365 | 0.0025 | 5 | 0.7844 | 0.0208 | 0.9254 | 0.0123 | 0.9872 | 0.0049 |
| 16 | mlp | Indoor+Outdoor | 0.0743 | 0.0028 | 5 | 0.3862 | 0.0158 | 0.7662 | 0.0163 | 0.9656 | 0.0075 |
| 16 | graph | Indoor+Outdoor | 0.0768 | 0.0043 | 5 | 0.3172 | 0.0207 | 0.7509 | 0.0389 | 0.9770 | 0.0043 |
| 16 | tcn | Indoor+Outdoor | 0.0743 | 0.0024 | 5 | 0.3895 | 0.0120 | 0.7661 | 0.0119 | 0.9664 | 0.0084 |
