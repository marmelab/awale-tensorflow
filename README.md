# awale-tensorflow

## Install

```
make install
```

Prerequisite: Python3 + openCV

## Run

Video:
```
make run -- -o video
```

Image:
```
make run -- -o image
```

Add `-s` for saving image without border
Add `-n` for directory name

Tensorflow:
```
make run -- -t
make run -- -a
make run -- -r
```
Use `-t` for training neural network
Use `-a` to display accuracy
Use `-r` to run neural network on specific folder
