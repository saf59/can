use actix_multipart::Multipart;
use actix_web::middleware::Logger;
use actix_web::{web, App, Error, HttpResponse, HttpServer};
use futures_util::stream::StreamExt as _;
use std::net::Ipv4Addr;

// Embedded OpenAPI spec and Swagger UI HTML
const OPENAPI_YAML: &str = include_str!("../static/openapi.yaml");
const SWAGGER_UI_HTML: &str = include_str!("../static/swagger_ui.html");
const OPENAPI_UI_HTML: &str = include_str!("../static/openapi_ui.html");

// Handler for Swagger UI
async fn swagger_ui() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/html")
        .body(SWAGGER_UI_HTML)
}
async fn openapi_ui() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/html")
        .body(OPENAPI_UI_HTML)
}

// Handler for OpenAPI spec
async fn api_docs() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/yaml")
        .body(OPENAPI_YAML)
}
// Main detection handler
async fn detect3(mut payload: Multipart) -> Result<HttpResponse, Error> {
    let mut frequency = None;
    let mut file_data = Vec::new();
    let mut signature = None;

    // Process multipart form data
    while let Some(item) = payload.next().await {
        let mut field = item?;
        let content_disposition = field.content_disposition();
        let field_name = content_disposition
            .expect("REASON")
            .get_name()
            .unwrap_or_default();

        match field_name {
            "frequency" => {
                let data = field.next().await.unwrap()?;
                let freq_str = String::from_utf8(data.to_vec()).unwrap_or("0".to_string());
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
                let sig_str = String::from_utf8(data.to_vec()).unwrap_or("0".to_string());
                let sig = sig_str.parse::<i64>().map_err(|e| {
                    actix_web::error::ErrorBadRequest(format!("Invalid signature: {}", e))
                })?;
                signature = Some(sig);
            }
            _ => {}
        }
    }
    // Validate required fields
    let freq = frequency.ok_or(actix_web::error::ErrorBadRequest("Missing frequency"))?;
    let sig = signature.ok_or(actix_web::error::ErrorBadRequest("Missing signature"))?;
    if file_data.is_empty() {
        return Err(actix_web::error::ErrorBadRequest("Missing file"));
    }

    // Example signature validation
    if (sig/137)*137 != sig {
        return Ok(HttpResponse::Forbidden().body("Invalid signature"));
    }
    let file_sig = crc32fast::hash(&file_data) * 137;

    if (file_sig as i64) != sig {
        return Ok(HttpResponse::Forbidden().body("Invalid signature"));
    }
    println!("Frequency: {:?}", &freq);
    println!("File sig: {}", file_sig);
    println!("Signature: {:?}", &sig);

    //detect()

    // Process your detection logic here (dummy response)
    Ok(HttpResponse::Ok().content_type("text/plain").body("-0.2"))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .wrap(Logger::default())
            // Register routes
            .service(web::resource("/detect3").route(web::post().to(detect3)))
            .route("/api-docs", web::get().to(api_docs))
            .route("/openapi-ui", web::get().to(openapi_ui))
            // Swagger UI
            .route("/", web::get().to(swagger_ui))
            .route("/index.html", web::get().to(swagger_ui))
            .route("/swagger-ui", web::get().to(swagger_ui))
    })
    .bind((Ipv4Addr::UNSPECIFIED, 9447))
    .unwrap()
    .run()
    .await
}
