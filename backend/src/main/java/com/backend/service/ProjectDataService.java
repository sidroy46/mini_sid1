package com.backend.service;

import com.backend.dto.MarkAttendanceResponse;
import com.backend.model.AttendanceRecord;
import com.backend.model.Student;
import com.backend.model.Subject;
import com.backend.repository.AttendanceRepository;
import com.backend.repository.StudentRepository;
import com.backend.repository.SubjectRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Service
public class ProjectDataService {

    private final StudentRepository studentRepository;
    private final SubjectRepository subjectRepository;
    private final AttendanceRepository attendanceRepository;

    private final Path faceUploadDir = Paths.get("uploads", "faces");
    private final Path faceAiDatasetDir = Paths.get("..", "face-ai", "dataset");
    private String faceRecognizeUrl;
    private String faceReloadUrl;

    @org.springframework.beans.factory.annotation.Value("${FACE_AI_HOST:http://localhost:5000}")
    public void setFaceAiHost(String host) {
        String base = host.startsWith("http") ? host : "https://" + host;
        this.faceRecognizeUrl = base + "/recognize";
        this.faceReloadUrl = base + "/reload-faces";
    }
    private final HttpClient httpClient = HttpClient.newHttpClient();

    public ProjectDataService(
            StudentRepository studentRepository,
            SubjectRepository subjectRepository,
            AttendanceRepository attendanceRepository
    ) {
        this.studentRepository = studentRepository;
        this.subjectRepository = subjectRepository;
        this.attendanceRepository = attendanceRepository;
    }

    public List<Student> getStudents() {
        return studentRepository.findAll().stream()
                .sorted(Comparator.comparing(Student::getId))
                .toList();
    }

    @Transactional
    public Student createStudent(String name, String rollNumber, String email, String department, List<MultipartFile> faceImages) throws IOException {
        validateStudent(name, rollNumber, email, department, null);

        Student student = new Student();
        student.setName(name.trim());
        student.setRollNumber(rollNumber.trim());
        student.setEmail(email.trim());
        student.setDepartment(department.trim());
        student.setFaceImagePaths(saveFaceImages(student.getRollNumber(), faceImages));

        return studentRepository.save(student);
    }

    @Transactional
    public Student updateStudent(Long id, String name, String rollNumber, String email, String department, List<MultipartFile> faceImages) throws IOException {
        Student existing = studentRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Student not found"));

        validateStudent(name, rollNumber, email, department, id);

        existing.setName(name.trim());
        existing.setRollNumber(rollNumber.trim());
        existing.setEmail(email.trim());
        existing.setDepartment(department.trim());
        if (faceImages != null && !faceImages.isEmpty()) {
            existing.setFaceImagePaths(saveFaceImages(existing.getRollNumber(), faceImages));
        }

        return studentRepository.save(existing);
    }

    @Transactional
    public void deleteStudent(Long id) {
        if (!studentRepository.existsById(id)) {
            throw new IllegalArgumentException("Student not found");
        }
        attendanceRepository.deleteByStudentId(id);
        studentRepository.deleteById(id);
    }

    public List<Subject> getSubjects() {
        return subjectRepository.findAll().stream()
                .sorted(Comparator.comparing(Subject::getId))
                .toList();
    }

    @Transactional
    public Subject createSubject(Subject request) {
        validateSubject(request, null);

        Subject subject = new Subject();
        subject.setCode(request.getCode().trim());
        subject.setName(request.getName().trim());
        subject.setFacultyName(request.getFacultyName().trim());
        subject.setClassStartTime(request.getClassStartTime());
        subject.setClassEndTime(request.getClassEndTime());
        return subjectRepository.save(subject);
    }

    @Transactional
    public Subject updateSubject(Long id, Subject request) {
        Subject existing = subjectRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Subject not found"));

        validateSubject(request, id);

        existing.setCode(request.getCode().trim());
        existing.setName(request.getName().trim());
        existing.setFacultyName(request.getFacultyName().trim());
        existing.setClassStartTime(request.getClassStartTime());
        existing.setClassEndTime(request.getClassEndTime());
        return subjectRepository.save(existing);
    }

    @Transactional
    public void deleteSubject(Long id) {
        if (!subjectRepository.existsById(id)) {
            throw new IllegalArgumentException("Subject not found");
        }
        attendanceRepository.deleteBySubjectId(id);
        subjectRepository.deleteById(id);
    }

    @Transactional
    public MarkAttendanceResponse markAttendance(Long subjectId, String imageData) {
        MarkAttendanceResponse response = new MarkAttendanceResponse();

        if (subjectId == null) {
            response.setStatus("failed");
            response.setMessage("Subject is required");
            return response;
        }

        if (imageData == null || imageData.isBlank()) {
            response.setStatus("failed");
            response.setMessage("Image is required");
            return response;
        }

        Subject subject = subjectRepository.findById(subjectId).orElse(null);
        if (subject == null) {
            response.setStatus("failed");
            response.setMessage("Subject not found");
            return response;
        }

        Map<String, Object> recognitionResult = recognizeFace(imageData);
        String recognitionStatus = String.valueOf(recognitionResult.getOrDefault("status", "fail"));

        if (!"success".equalsIgnoreCase(recognitionStatus)) {
            response.setStatus("failed");
            response.setMessage(String.valueOf(recognitionResult.getOrDefault("message", "Face Not Recognized")));
            Object confidence = recognitionResult.get("confidence");
            if (confidence != null) {
                response.setConfidence(confidence.toString());
            }
            return response;
        }

        String matchedLabel = String.valueOf(recognitionResult.getOrDefault("name", "")).trim();
        Optional<Student> matched = studentRepository.findAll().stream()
                .filter(student -> student.getRollNumber().equalsIgnoreCase(matchedLabel)
                        || student.getName().equalsIgnoreCase(matchedLabel))
                .findFirst();

        if (matched.isEmpty()) {
            response.setStatus("failed");
            response.setMessage("Recognized face does not match any registered student");
            return response;
        }

        Student student = matched.get();
        LocalDate today = LocalDate.now();

        boolean exists = attendanceRepository.existsByStudentIdAndSubjectIdAndDate(student.getId(), subjectId, today);
        if (exists) {
            response.setStatus("failed");
            response.setMessage("Attendance already marked for this student and subject today");
            return response;
        }

        AttendanceRecord row = new AttendanceRecord();
        row.setStudentId(student.getId());
        row.setSubjectId(subject.getId());
        row.setStudentName(student.getName());
        row.setRollNumber(student.getRollNumber());
        row.setSubjectName(subject.getName());
        row.setFacultyName(subject.getFacultyName());
        row.setDate(today);
        row.setTime(LocalTime.now().withNano(0));

        AttendanceRecord saved = attendanceRepository.save(row);

        response.setStatus("success");
        response.setMessage("Attendance marked");
        response.setAttendanceId(saved.getId());
        response.setName(saved.getStudentName());
        response.setSubject(saved.getSubjectName());
        response.setDate(saved.getDate().toString());
        response.setTime(saved.getTime().toString());
        Object confidence = recognitionResult.get("confidence");
        response.setConfidence(confidence == null ? "N/A" : confidence.toString());
        return response;
    }

    public List<AttendanceRecord> getDailyReport(LocalDate date) {
        return attendanceRepository.findByDateOrderByTimeDesc(date);
    }

    public List<AttendanceRecord> getMonthlyReport(int year, int month) {
        LocalDate from = LocalDate.of(year, month, 1);
        LocalDate to = from.withDayOfMonth(from.lengthOfMonth());
        return attendanceRepository.findByDateBetweenOrderByDateDescTimeDesc(from, to);
    }

    public List<AttendanceRecord> getStudentReport(Long studentId) {
        return attendanceRepository.findByStudentIdOrderByDateDescTimeDesc(studentId);
    }

    public Map<String, Object> getDashboardSummary() {
        int totalStudents = (int) studentRepository.count();
        int todayAttendance = (int) attendanceRepository.countByDate(LocalDate.now());
        int percentage = totalStudents == 0 ? 0 : (int) Math.round((todayAttendance * 100.0) / totalStudents);

        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("totalStudents", totalStudents);
        summary.put("todayAttendance", todayAttendance);
        summary.put("attendancePercentage", percentage);
        return summary;
    }

    public Map<String, Integer> getDashboardChart() {
        LocalDate today = LocalDate.now();
        DateTimeFormatter formatter = DateTimeFormatter.ISO_DATE;
        Map<String, Integer> chart = new LinkedHashMap<>();

        for (int i = 6; i >= 0; i--) {
            LocalDate target = today.minusDays(i);
            int count = (int) attendanceRepository.countByDate(target);
            chart.put(formatter.format(target), count);
        }
        return chart;
    }

    private void validateStudent(String name, String rollNumber, String email, String department, Long updatingId) {
        if (isBlank(name) || isBlank(rollNumber) || isBlank(email) || isBlank(department)) {
            throw new IllegalArgumentException("All student fields are required");
        }

        String normalizedRoll = rollNumber.trim();
        String normalizedEmail = email.trim();

        boolean duplicateRoll = updatingId == null
                ? studentRepository.existsByRollNumberIgnoreCase(normalizedRoll)
                : studentRepository.existsByRollNumberIgnoreCaseAndIdNot(normalizedRoll, updatingId);
        if (duplicateRoll) {
            throw new IllegalArgumentException("Roll number already exists");
        }

        boolean duplicateEmail = updatingId == null
                ? studentRepository.existsByEmailIgnoreCase(normalizedEmail)
                : studentRepository.existsByEmailIgnoreCaseAndIdNot(normalizedEmail, updatingId);
        if (duplicateEmail) {
            throw new IllegalArgumentException("Email already exists");
        }
    }

    private void validateSubject(Subject request, Long updatingId) {
        if (request == null || isBlank(request.getCode()) || isBlank(request.getName()) || isBlank(request.getFacultyName())
                || isBlank(request.getClassStartTime()) || isBlank(request.getClassEndTime())) {
            throw new IllegalArgumentException("All subject fields are required");
        }

        String normalizedCode = request.getCode().trim();
        boolean duplicateCode = updatingId == null
                ? subjectRepository.existsByCodeIgnoreCase(normalizedCode)
                : subjectRepository.existsByCodeIgnoreCaseAndIdNot(normalizedCode, updatingId);
        if (duplicateCode) {
            throw new IllegalArgumentException("Subject code already exists");
        }
    }

    private List<String> saveFaceImages(String rollNumber, List<MultipartFile> faceImages) throws IOException {
        if (faceImages == null || faceImages.isEmpty()) {
            return new ArrayList<>();
        }

        Files.createDirectories(faceUploadDir);
        Path datasetStudentDir = faceAiDatasetDir.resolve(rollNumber);
        Files.createDirectories(datasetStudentDir);
        List<String> stored = new ArrayList<>();

        for (int i = 0; i < Math.min(faceImages.size(), 3); i++) {
            MultipartFile file = faceImages.get(i);
            if (file == null || file.isEmpty()) {
                continue;
            }
            String filename = rollNumber + "_" + System.currentTimeMillis() + "_" + i + ".jpg";
            Path target = faceUploadDir.resolve(filename);
            Files.copy(file.getInputStream(), target, StandardCopyOption.REPLACE_EXISTING);
            Path datasetTarget = datasetStudentDir.resolve(filename);
            Files.copy(target, datasetTarget, StandardCopyOption.REPLACE_EXISTING);
            stored.add(target.toString().replace('\\', '/'));
        }

        reloadKnownFaces();

        return stored;
    }

    private Map<String, Object> recognizeFace(String imageData) {
        try {
            String payload = "{\"image\":\"" + escapeJson(imageData) + "\"}";
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(faceRecognizeUrl))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(payload))
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                return Map.of("status", "fail", "message", "Face service error: HTTP " + response.statusCode());
            }

            String body = response.body();
            String status = extractJsonString(body, "status");
            String message = extractJsonString(body, "message");
            String name = extractJsonString(body, "name");
            String confidence = extractJsonNumberOrString(body, "confidence");

            Map<String, Object> parsed = new LinkedHashMap<>();
            parsed.put("status", status == null ? "fail" : status);
            if (message != null) {
                parsed.put("message", message);
            }
            if (name != null) {
                parsed.put("name", name);
            }
            if (confidence != null) {
                parsed.put("confidence", confidence);
            }
            return parsed;
        } catch (Exception exception) {
            return Map.of("status", "fail", "message", "Face service unavailable");
        }
    }

    private void reloadKnownFaces() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(faceReloadUrl))
                    .POST(HttpRequest.BodyPublishers.noBody())
                    .build();
            httpClient.send(request, HttpResponse.BodyHandlers.discarding());
        } catch (Exception ignored) {
        }
    }

    private String escapeJson(String input) {
        return input
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "")
                .replace("\r", "");
    }

    private String extractJsonString(String json, String key) {
        Pattern pattern = Pattern.compile("\\\"" + Pattern.quote(key) + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
        Matcher matcher = pattern.matcher(json);
        return matcher.find() ? matcher.group(1) : null;
    }

    private String extractJsonNumberOrString(String json, String key) {
        Pattern pattern = Pattern.compile("\\\"" + Pattern.quote(key) + "\\\"\\s*:\\s*(\\\"([^\\\"]*)\\\"|[-+]?\\d*\\.?\\d+)");
        Matcher matcher = pattern.matcher(json);
        if (!matcher.find()) {
            return null;
        }
        String raw = matcher.group(1);
        if (raw.startsWith("\"") && raw.endsWith("\"")) {
            return raw.substring(1, raw.length() - 1);
        }
        return raw;
    }

    private boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }
}
