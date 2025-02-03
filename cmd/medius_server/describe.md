Okay, I need to generate a Rust server based on the provided OpenAPI specification. Let me start by understanding the requirements from the YAML file. The main endpoint is /detect3 which accepts a POST request with multipart/form-data containing frequency, file, and signature. The server also needs to have a Swagger UI endpoint.

First, I should choose the appropriate Rust frameworks. For the web server, Actix-web is a good choice because it's widely used and supports asynchronous operations. To handle OpenAPI and generate the Swagger UI, I can use the paperclip crate, which integrates with Actix-web and provides OpenAPI support.

Looking at the YAML, the /detect3 endpoint requires three form fields: frequency (a double), file (a binary WAV file), and signature (an int64). In Actix, handling multipart forms can be done using web::Form, but for files, I might need to use actix-multipart. Wait, maybe I should use Multipart directly to parse the form data. Let me check how actix-multipart works.

For the response, the endpoint returns a plain text number on success (200), and various error codes. The error handling needs to map application errors to the correct HTTP statuses like 400, 403, etc.

Including Swagger UI means adding a route that serves the UI. Paperclip can help generate the OpenAPI spec dynamically, but since the user already has the spec, maybe they want to serve the existing openapi.yaml? Or generate it from the code. Wait, the user mentioned generating a server with an endpoint for Swagger UI. The Swagger UI typically needs the OpenAPI JSON or YAML file to be served, and then the UI can fetch it. So perhaps the server should have a route like /api-docs that serves the openapi.yaml content, and another route like /swagger-ui that serves the HTML for the UI.

Alternatively, using paperclip's plugin might automatically set up the Swagger UI. Let me look up how paperclip's Actix integration works. Paperclip can annotate the routes and generate the OpenAPI spec, then provide a UI endpoint. However, the user's OpenAPI spec is already defined, so maybe we need to serve that static file. Hmm, this might be a bit conflicting. The user might expect the server to both implement the endpoints and serve the Swagger UI based on the provided spec.

Alternatively, perhaps the generated server should use the existing openapi.yaml to configure the Swagger UI. But in practice, Swagger UI is usually served with a HTML page that references the OpenAPI spec URL. So the server needs to serve two things: the OpenAPI spec (probably at /api-docs) and the Swagger UI at another path, like /swagger.

So steps I need to take:

1. Set up Actix-web server with necessary dependencies.

2. Create a handler for POST /detect3 that processes multipart form data.

3. Implement error handling for different response statuses.

4. Serve the OpenAPI spec (maybe embedded as a static file) at /api-docs.

5. Serve Swagger UI HTML that points to /api-docs.

For the multipart handling, I'll need to use actix-multipart. Let's structure the handler to parse the three fields: frequency, file, and signature. Each part of the multipart form needs to be processed in sequence. Then, validate each field: frequency is a number, signature is an integer, and the file is a valid WAV (though actual validation might be minimal in this example).

For the Swagger UI, I can include the Swagger UI static files or serve a simple HTML page that loads the Swagger UI from a CDN and points to the local /api-docs endpoint. That would be easier than embedding the files.

Now, putting this into code:

Dependencies in Cargo.toml:

- actix-web = "4"

- actix-multipart = "0.6"

- paperclip = { version = "0.8", features = ["actix4", "v3"] }

- serde = { version = "1.0", features = ["derive"] }

- futures-util = "0.3" for async streams

- mime = "0.3"

Then, define the main function with Actix app setup. Use paperclip's OpenApi plugin to auto-generate the spec, but since we have a static spec, maybe it's easier to serve the existing openapi.yaml. Wait, the user provided the openapi.yaml content, but in the server, perhaps we should serve that content at /api-docs. So, create a handler that returns the YAML content with the correct content type.

But the Swagger UI needs to access the OpenAPI spec. So, adding a route for /api-docs that returns the YAML file. Then, another route for /swagger-ui that serves an HTML page which is the Swagger UI, configured to load the /api-docs URL.

For simplicity, the HTML can be a simple template that includes the Swagger UI assets from a CDN, like:

<!DOCTYPE html>

<html>

<head>

<title>Swagger UI</title>

<link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3/swagger-ui.css">

</head>

<body>

<div id="swagger-ui"></div>

<script src="https://unpkg.com/swagger-ui-dist@3/swagger-ui-bundle.js"></script>

<script>

window.onload = function() {

SwaggerUIBundle({

url: "/api-docs",

dom_id: '#swagger-ui',

});

};

</script>

</body>

</html>

Then, the server can serve this HTML at /swagger-ui.

So, the server will have:

- GET /swagger-ui: serves the HTML

- GET /api-docs: serves the openapi.yaml content

- POST /detect3: processes the multipart form

Now, the handler for /detect3. Using actix-multipart's Multipart middleware. The handler will need to asynchronously process each part of the form. For each field, check the name and parse the content.

For example:

Loop through each part in the multipart stream. For each part, check if the name is "frequency", "file", or "signature". Parse the content accordingly. Frequency is a number, signature is an integer, file is saved as bytes.

Once all parts are processed, validate them. If any part is missing, return 400. If the signature is invalid (some check, but the example just uses 123456789), return 403. Then, return a 200 with a text/plain response of a number like -0.2.

But in the example, the response is a plain number, so the handler should return a String with the number and set the content type to text/plain.

Implementing error handling: create custom error types that can be converted to Actix responses with appropriate status codes. For example, missing field -> 400, invalid signature -> 403, etc.

Putting it all together, the code would look something like:

main.rs:

use actix_web::{web, App, HttpResponse, HttpServer, Responder, Error};

use actix_multipart::Multipart;

use futures_util::stream::StreamExt as _;

use serde::{Deserialize, Serialize};

use paperclip::actix::{web::ServiceConfig, OpenApiExt};

async fn swagger_ui() -> HttpResponse {

HttpResponse::Ok()

.content_type("text/html")

.body(include_str!("swagger_ui.html"))

}

async fn api_docs() -> HttpResponse {

HttpResponse::Ok()

.content_type("text/yaml")

.body(include_str!("openapi.yaml"))

}

async fn detect3(mut payload: Multipart) -> Result<HttpResponse, Error> {

let mut frequency = None;

let mut file_data = Vec::new();

let mut signature = None;

while let Some(item) = payload.next().await {

let mut field = item?;

let content_disposition = field.content_disposition();

let name = content_disposition.get_name().unwrap_or_default();

match name {

"frequency" => {

let data = field.next().await.unwrap()?;

let freq_str = String::from_utf8(data.to_vec()).map_err(|e| {

actix_web::error::ErrorBadRequest(format!("Invalid frequency: {}", e))

})?;

let freq = freq_str.parse::<f64>().map_err(|e| {

actix_web::error::ErrorBadRequest(format!("Invalid frequency: {}", e))

})?;

frequency = Some(freq);

}

"file" => {

while let Some(chunk) = field.next().await {

let data = chunk?;

file_data.extend_from_slice(&data);

}

}

"signature" => {

let data = field.next().await.unwrap()?;

let sig_str = String::from_utf8(data.to_vec()).map_err(|e| {

actix_web::error::ErrorBadRequest(format!("Invalid signature: {}", e))

})?;

let sig = sig_str.parse::<i64>().map_err(|e| {

actix_web::error::ErrorBadRequest(format!("Invalid signature: {}", e))

})?;

signature = Some(sig);

}

_ => {}

}

}

if frequency.is_none() || file_data.is_empty() || signature.is_none() {

return Ok(HttpResponse::BadRequest().body("Missing required fields"));

}

// Example signature check

if signature.unwrap() != 123456789 {

return Ok(HttpResponse::Forbidden().body("Invalid signature"));

}

// Process the file and frequency here (dummy response)

Ok(HttpResponse::Ok()

.content_type("text/plain")

.body("-0.2"))

}

#[actix_web::main]

async fn main() -> std::io::Result<()> {

HttpServer::new(|| {

App::new()

.wrap(paperclip::actix::OpenApiExt::new())

.service(

web::resource("/detect3")

.route(web::post().to(detect3))

)

.route("/api-docs", web::get().to(api_docs))

.route("/swagger-ui", web::get().to(swagger_ui))

})

.bind("localhost:8080")?

.run()

.await

}

But I need to include the openapi.yaml content and the swagger_ui.html. However, in the code, I used include_str! to embed them. So, in the Rust project, there should be a directory structure with these files.

Also, the server is bound to https://localhost:8080 as per the OpenAPI spec. But Actix by default uses HTTP, not HTTPS. The user might need to set up TLS, but that's beyond the basic setup. The example uses HTTP.

Wait, the server URL in the YAML is https://localhost:8080, but the generated code uses HTTP. To handle HTTPS, additional configuration with .bind_openssl or similar is needed, but for simplicity, the answer can note that HTTPS setup is required separately.

Another thing: The OpenAPI spec in the YAML is served at /api-docs, which matches the server's route. The Swagger UI is served at /swagger-ui, which is not part of the original spec, but the user requested an endpoint for the Swagger UI.

Potential issues:

- The multipart handling may not correctly parse the fields. For example, the "file" part is read as a string in the YAML, but in reality, it's a binary. However, in the code, the file is read as bytes and stored in a Vec<u8>.

- The example uses a dummy check for the signature (comparing to 123456789), which aligns with the example in the YAML.

- The error responses for 422 and 500 are not implemented, but the handler returns 400 and 403. To fully comply, more error variants would be needed, but for brevity, the example includes basic error handling.

Thus, the generated code provides a basic structure that can be expanded with actual processing logic and more detailed error handling.