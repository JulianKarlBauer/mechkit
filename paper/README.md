# Build pdf locally
- install docker
- navigate to directory which contains paper.md

```
sudo docker run --rm \
    --volume $PWD:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/paperdraft
```