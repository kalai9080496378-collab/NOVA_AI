import enum


class AccountStatus(str, enum.Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class InputLanguage(str, enum.Enum):
    EN = "en"
    TA = "ta"
    THANGLISH = "thanglish"


class VisualStyle(str, enum.Enum):
    REALISTIC = "realistic"
    CARTOON_3D = "cartoon_3d"
    ANIME = "anime"


class StoryStatus(str, enum.Enum):
    STORY_SUBMITTED = "STORY_SUBMITTED"
    STORY_ANALYSIS = "STORY_ANALYSIS"
    SCENE_PLANNING = "SCENE_PLANNING"
    SCENES_GENERATED = "SCENES_GENERATED"
    GENERATING_SCENES = "GENERATING_SCENES"
    ASSEMBLING_VIDEO = "ASSEMBLING_VIDEO"
    READY_FOR_PREVIEW = "READY_FOR_PREVIEW"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SceneGenerationStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModerationStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED = "flagged"


class VideoVisibility(str, enum.Enum):
    PUBLIC = "public"
    UNLISTED = "unlisted"
    PRIVATE = "private"


class TargetType(str, enum.Enum):
    VIDEO = "video"
    COMMENT = "comment"
    USER = "user"


class ReportStatus(str, enum.Enum):
    PENDING = "pending"
    REVIEWED = "reviewed"
    DISMISSED = "dismissed"
    ACTION_TAKEN = "action_taken"
