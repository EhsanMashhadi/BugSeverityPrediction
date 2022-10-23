package data.method_extractor;

public class BugsJarBug extends Bug {
    private String projectVersion;

    public BugsJarBug(String projectName, String projectVersion, String severity) {
        super(projectName, severity);
        this.projectVersion = projectVersion;
    }

    public String getProjectVersion() {
        return projectVersion;
    }
}
