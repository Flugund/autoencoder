### Auto Encoder

This project aims to solve the MNIST handwritten digit problem defined here: http://yann.lecun.com/exdb/mnist/

### How to run

```
cargo run -r
```

### Enviroment variables

```
PRELOAD_NETWORK="./data/networks/<filename>.json" cargo run -r
```

### Example output

The latest run of this program yeilded the following result:

```
2023-08-05T22:25:57.189 [INFO] - Completed training
2023-08-05T22:25:57.190 [INFO] - Network trained with training data
2023-08-05T22:26:18.792 [INFO] - Right: 9991.0, Wrong: 9.0, Percent: 99.91%, Failed: [0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.0, 1.0, 1.0, 1.0]
2023-08-05T22:26:18.792 [INFO] - Running final test...
2023-08-05T22:26:40.420 [INFO] - Right: 9812.0, Wrong: 188.0, Percent: 98.11999999999999%, Failed: [7.0, 10.0, 19.0, 20.0, 18.0, 22.0, 15.0, 25.0, 23.0, 29.0]
```
