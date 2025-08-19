package com.example.heart.controller;

import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

@RestController
@RequestMapping("/api")
public class DiagnosisController {

    private final RestTemplate restTemplate = new RestTemplate();

    @PostMapping("/diagnosis")
    public ResponseEntity<String> proxyDiagnosis(@RequestBody Map<String, Object> payload) {
        // URL ke FastAPI
        String fastApiUrl = "http://127.0.0.1:8000/diagnose";

        // kirim request ke FastAPI
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(payload, headers);

        ResponseEntity<String> response = restTemplate.exchange(
                fastApiUrl,
                HttpMethod.POST,
                requestEntity,
                String.class
        );

        return ResponseEntity.status(response.getStatusCode()).body(response.getBody());
    }
}
