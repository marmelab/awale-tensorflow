<table>
        <tr>
            <td><img width="120" src="https://cdnjs.cloudflare.com/ajax/libs/octicons/8.5.0/svg/rocket.svg" alt="onboarding" /></td>
            <td><strong>Archived Repository</strong><br />
            The code of this repository was written during a <a href="https://marmelab.com/blog/2018/09/05/agile-integration.html">Marmelab agile integration</a>. It illustrates the efforts of a new hiree, who had to implement a board game in several languages and platforms as part of his initial learning. Some of these efforts end up in failure, but failure is part of our learning process, so the code remains publicly visible.<br />
        <strong>This code is not intended to be used in production, and is not maintained.</strong>
        </td>
        </tr>
</table>

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
Use `-r` to display number of pebble in images
