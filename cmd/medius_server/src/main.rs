use std::env;
use actix_multipart::Multipart;
use actix_web::middleware::Logger;
use actix_web::{web, App, Error, HttpRequest, HttpResponse, HttpServer};
use futures_util::stream::StreamExt as _;
use std::net::Ipv4Addr;
use env_logger::Env;
use log::info;
use medius_utils::detect;

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
async fn openapi_ui(req: HttpRequest) -> HttpResponse {
    let runtime = real_body(req, OPENAPI_UI_HTML);
    HttpResponse::Ok()
        .content_type("text/html")
        .body(runtime)
}

// Handler for OpenAPI spec
async fn api_docs(req: HttpRequest) -> HttpResponse {
    let runtime = real_body(req, OPENAPI_YAML);
    HttpResponse::Ok()
        .content_type("text/yaml")
        .body(runtime)
}
fn real_body(http_request: HttpRequest, src: &str) -> String {
    let info = http_request.connection_info();
    let runtime = format!("{}://{}", info.scheme(),info.host());
    src.replace("http://localhost:8080", &runtime)
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
                if (sig/137)*137 != sig {
                    return Err(actix_web::error::ErrorForbidden("Invalid signature"));
                }
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
    if freq < 10000.0 {
        return Err(actix_web::error::ErrorBadRequest("Frequency is wrong"));
    }
    // Signature validation
    let file_sig:i64 = (crc32fast::hash(&file_data) as i64) * 137_i64;
    if file_sig != sig {
        info!("Frequency: {:?}, file sig: {}, signature: {:?}", &freq,file_sig,&sig);
        return Err(actix_web::error::ErrorForbidden("Invalid signature"));
    }
    let wd = match detect(&file_data, freq as f32, false) {
        Ok(dist) => dist.to_string(),
        Err(e) => {
            info!("Frequency: {:?}, file sig: {}, signature: {:?}", &freq,file_sig,&sig);
            return Err(actix_web::error::ErrorBadRequest(e.to_string()));
        }
    };
    // info!("Detection result: {}", wd);
    Ok(HttpResponse::Ok().content_type("text/plain").body(wd))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();
    let port = env::var("MEDIUS_PORT").unwrap_or_else(|_| "9447".to_string()).parse().unwrap();
    HttpServer::new(|| {
        App::new()
            // Register routes
            .service(web::resource("/detect3").route(web::post().to(detect3)))
            .route("/api-docs", web::get().to(api_docs))
            .route("/openapi-ui", web::get().to(openapi_ui))
            // Swagger UI
            .route("/", web::get().to(swagger_ui))
            .route("/index.html", web::get().to(swagger_ui))
            .route("/swagger-ui", web::get().to(swagger_ui))
            .wrap(Logger::default())
    })
    .bind((Ipv4Addr::UNSPECIFIED, port))
    .unwrap()
    .run()
    .await
}
