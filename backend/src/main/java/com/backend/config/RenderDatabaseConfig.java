package com.backend.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.boot.jdbc.DataSourceBuilder;
import org.springframework.core.env.Environment;

import javax.sql.DataSource;

@Configuration
public class RenderDatabaseConfig {

    @Bean
    @Primary
    public DataSource dataSource(Environment env) {
        // Render literally always injects "RENDER=true" as a system environment variable.
        String isRender = env.getProperty("RENDER");
        
        if ("true".equals(isRender)) {
            return DataSourceBuilder.create()
                    .url("jdbc:postgresql://dpg-d75li5kr85hc73cos150-a/faceattendence_db")
                    .username("faceattendence_db_user")
                    .password("THBtE0vrM5ZztSzAQhcTLGEyemkdMr0P")
                    .driverClassName("org.postgresql.Driver")
                    .build();
        }

        // If not running in Render cloud (i.e. running on your laptop), 
        // fall back perfectly to the original MySQL local database configuration.
        String fallbackUrl = env.getProperty("spring.datasource.url", "jdbc:mysql://localhost:3306/attendance_system?createDatabaseIfNotExist=true&useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=UTC");
        String fallbackUsername = env.getProperty("spring.datasource.username", "root");
        String fallbackPassword = env.getProperty("spring.datasource.password", "Root@1234");
        
        return DataSourceBuilder.create()
                .url(fallbackUrl)
                .username(fallbackUsername)
                .password(fallbackPassword)
                .driverClassName("com.mysql.cj.jdbc.Driver")
                .build();
    }
}
