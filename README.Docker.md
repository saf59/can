### Building and running your application
It must be ROOT of the application
Add "python3" to the Dockerfile with RUN apk add:
	RUN apk add --no-cache clang lld musl-dev git python3
Remove src dir and add right dirs - cmd,crates,models to RUN --mount rows: 
	RUN --mount=type=bind,source=cmd,target=cmd \
    --mount=type=bind,source=crates,target=crates \
	--mount=type=bind,source=models,target=models \
Add --bin to cargo build:
	cargo build --bin $APP_NAME


When you're ready, start your application by running:
`docker compose up --build`.

Your application will be available at http://localhost:9447.

### Deploying your application to the cloud

First, build your image, e.g.: `docker build -t meds .`.
If your cloud uses a different CPU architecture than your development
machine (e.g., you are on a Mac M1 and your cloud provider is amd64),
you'll want to build the image for that platform, e.g.:
`docker build --platform=linux/amd64 -t myapp .`.

Then, push it to your registry, e.g.
'''
docker login
docker tag can-server alsh/meds
or
docker tag meds alsh/meds
docker push alsh/meds

Consult Docker's [getting started](https://docs.docker.com/go/get-started-sharing/)
docs for more detail on building and pushing.

### References
* [Docker's Rust guide](https://docs.docker.com/language/rust/)

## detect3 ##
1) чтобы работало быстро, каталоги монтируем, но в .dockerignore нужное надо разрешить
2) best3 чтобы правильные model.meta model.safetensors
3) каталог models в .dockerignore, иначе собирает очень долго
4) при билде нужны model.meta model.safetensors - mount it, 
   а при тесте x1_y1.wav поэтому копируем их в каталог export
5) docker build -t detect3 -f Dockerfile_detect3 .
6) docker run detect3  
	-0.400
7) Где то:  
	docker image save detect3 -o detect3.zip
8) В архиве будет слой размером около 1-5M, а в нем еще один, а в нем \usr\bin\detect3 

## detect4 ##
аналог detect3, но best4 и тестовый файл in.wav и результат должен быть 0