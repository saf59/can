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
docker push alsh/meds

Consult Docker's [getting started](https://docs.docker.com/go/get-started-sharing/)
docs for more detail on building and pushing.

### References
* [Docker's Rust guide](https://docs.docker.com/language/rust/)