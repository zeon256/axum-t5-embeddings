## `axum-t5-embeddings`
> This project allows you to use axum to serve t5 models and use it to get embeddings for any given text.

## Why did I do this?
In terms of performance, there isn't that big of a difference compared to using a python
server that uses FastAPI. The main reason is just the ease of use. I don't like to worry about 
dependencies and virtual environments. I just want to run a single binary and be done with it.
This project allows me to do that.

## Tested models
- [hkunlp/instructor-large](https://huggingface.co/hkunlp/instructor-large)

## Build
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Usage
```bash
RUST_LOG="axum_t5_embeddings=debug,tower_http=debug" axum-t5-embeddings -w 4 -m PATH_TO_YOUR_MODEL
```
Then you can curl the server:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"inputs": ["Hello world!"]}' http://localhost:9999/feature-extraction
```

Output
```json
[
    [
        -0.036760102957487106,
        -0.006628172006458044,
        -0.015578031539916992,
        0.018409788608551025,
        -0.021647775545716286,
        0.02015158161520958,
        0.015129962004721165,
        -0.039880432188510895,
        0.02829710580408573,
        0.039527084678411484,
        0.034133102744817734,
        0.053229477256536484,
        -0.022237172350287437,
        0.04002070799469948,
        0.042418450117111206,
        0.04097423702478409,
        -0.06126898154616356,
        -0.0399135947227478,
        .
        .
        .
    ]
]
```

## Help
```bash
axum-t5-embeddings --help
```

Output 
```
Usage: axum-t5-embeddings [-l] [-p <port>] [-w <no-workers>] -m <model>

Axum-T5 server written in Rust

Options:
  -l, --listen      listen on 0.0.0.0
  -p, --port        port number
  -w, --no-workers  number of workers
  -m, --model       path to model
  --help            display usage information
```

## License
This project is licensed under the [MIT license](LICENSE).
