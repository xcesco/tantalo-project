# Portainer
È un gestore grafico per docker.

```shell
docker run -d -p 9001:9001 -p 9002:9002 --name portainer_agent --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v /var/lib/docker/volumes:/var/lib/docker/volumes portainer/agent:latest
```