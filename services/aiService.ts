
import { GoogleGenAI, Type } from "@google/genai";
import { ResumeData, CareerMatch, LearningPlan, QuizQuestion, ProjectAnalysis, PersonalityAnalysis, ScrapedJob, ATSAnalysis, MarketAnalysis } from "../types";

// Core Initialization - Exclusively using process.env.API_KEY for Tier 1 platform injection
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// Use stable model so all features work (Profile Init, Jobs, Trends, Evaluation, etc.)
const GEMINI_MODEL = "gemini-2.5-flash";
const GEMINI_MODEL_PRO = "gemini-2.5-pro";

/**
 * NEURAL QUOTA MONITOR
 * Cooldown UI disabled – no artificial delays or "cooling down" screen.
 */
const savedProMode = localStorage.getItem('ace_pro_mode') === 'true';

export const quotaState = {
  callsThisMinute: 0,
  isCoolingDown: false,
  cooldownRemaining: 0,
  lastReset: Date.now(),
  isProMode: savedProMode
};

let lastRequestTime = 0;
const MIN_GAP_MS = 10; // Small gap to avoid request overlap only

const throttle = async () => {
  if (Date.now() - quotaState.lastReset > 60000) {
    quotaState.callsThisMinute = 0;
    quotaState.lastReset = Date.now();
  }
  const now = Date.now();
  const timeSinceLast = now - lastRequestTime;
  if (timeSinceLast < MIN_GAP_MS) {
    await new Promise(resolve => setTimeout(resolve, MIN_GAP_MS - timeSinceLast));
  }
  lastRequestTime = Date.now();
};

export class QuotaError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "QuotaError";
  }
}

const MAX_RETRIES = 5;

async function handleGeminiCall<T>(call: () => Promise<T>, retries = MAX_RETRIES): Promise<T> {
  try {
    await throttle();
    const result = await call();
    quotaState.callsThisMinute++;
    return result;
  } catch (error: any) {
    const errorStr = JSON.stringify(error).toLowerCase();
    const isQuota = error.status === 429 || errorStr.includes('429') || errorStr.includes('quota') || errorStr.includes('exhausted');

    if (isQuota && retries > 0) {
      const wait = 200 * (MAX_RETRIES - retries + 1);
      console.debug(`[Neural Link] Rate limit hit, retrying in ${wait}ms...`);
      await new Promise(resolve => setTimeout(resolve, wait));
      return handleGeminiCall(call, retries - 1);
    }

    throw error;
  }
}

function safeExtractJson(text: string): any {
  try {
    const match = text.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
    if (match) return JSON.parse(match[0]);
    return JSON.parse(text);
  } catch (e) {
    console.error("JSON Extraction failed", text);
    throw new Error("AI response format error.");
  }
}

export const initiateProfile = async (resumeText: string): Promise<{ 
  resumeData: ResumeData, 
  analysis: ATSAnalysis, 
  rewritten: string 
}> => {
  return handleGeminiCall(async () => {
    const response = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: `You are an elite career strategist. Analyze the provided resume text.
      Extract full structured profile data, perform ATS audit, and rewrite into LaTeX-style Markdown.
      Resume text: ${resumeText}`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            resumeData: {
              type: Type.OBJECT,
              properties: {
                name: { type: Type.STRING },
                email: { type: Type.STRING },
                phone: { type: Type.STRING },
                location: { type: Type.STRING },
                skills: { type: Type.ARRAY, items: { type: Type.STRING } },
                experience: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      title: { type: Type.STRING },
                      company: { type: Type.STRING },
                      duration: { type: Type.STRING },
                      description: { type: Type.STRING },
                    },
                    required: ["title", "company", "duration", "description"],
                  },
                },
                education: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      degree: { type: Type.STRING },
                      institution: { type: Type.STRING },
                      year: { type: Type.STRING },
                    },
                    required: ["degree", "institution", "year"],
                  },
                },
              },
              required: ["name", "email", "skills", "experience", "education"],
            },
            analysis: {
              type: Type.OBJECT,
              properties: {
                ats_score: { type: Type.NUMBER },
                strengths: { type: Type.ARRAY, items: { type: Type.STRING } },
                improvements: { type: Type.ARRAY, items: { type: Type.STRING } },
                keywordGaps: { type: Type.ARRAY, items: { type: Type.STRING } },
                overallFeedback: { type: Type.STRING },
                action_verbs: { type: Type.ARRAY, items: { type: Type.STRING } },
                projects_to_add: { type: Type.ARRAY, items: { type: Type.STRING } },
              },
              required: ["ats_score", "strengths", "improvements", "keywordGaps", "overallFeedback", "action_verbs", "projects_to_add"],
            },
            rewrittenMarkdown: {
              type: Type.STRING,
            },
          },
          required: ["resumeData", "analysis", "rewrittenMarkdown"],
        },
      },
    });

    const parsed = JSON.parse(response.text || "{}");
    if (parsed.analysis && parsed.analysis.ats_score) {
      parsed.analysis.score = parsed.analysis.ats_score;
    }
    return {
      resumeData: parsed.resumeData,
      analysis: parsed.analysis,
      rewritten: parsed.rewrittenMarkdown
    };
  });
};

export const findJobMatches = async (resume: ResumeData | null, customQuery?: string): Promise<{ jobs: ScrapedJob[]; links: { title: string; url: string }[] }> => {
  return handleGeminiCall(async () => {
    const query = customQuery || (resume?.skills?.[0] || "Software Engineer");
    const response = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: `Suggest 5 realistic job openings for "${query}". Use your knowledge of real companies and roles. Return a JSON array of objects with: title, company, location, salary (e.g. "$120k-150k"), description (short), applyUrl (use placeholder like "https://linkedin.com/jobs/view/123" or company career page).`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            jobs: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  title: { type: Type.STRING },
                  company: { type: Type.STRING },
                  location: { type: Type.STRING },
                  salary: { type: Type.STRING },
                  description: { type: Type.STRING },
                  applyUrl: { type: Type.STRING },
                },
                required: ["title", "company", "location", "salary", "description", "applyUrl"],
              },
            },
          },
          required: ["jobs"],
        },
      },
    });

    const parsed = JSON.parse(response.text || "{}");
    const jobs = Array.isArray(parsed.jobs) ? parsed.jobs : (Array.isArray(parsed) ? parsed : []);
    return { jobs: jobs.slice(0, 5), links: [] };
  });
};

export const getMarketTrends = async (userSkills: string[]): Promise<MarketAnalysis> => {
  return handleGeminiCall(async () => {
    const skillsString = userSkills.join(", ");
    const response = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: `Based on current tech industry knowledge, analyze global job market trends relevant to these skills: ${skillsString}. Return a JSON object with: overallSentiment (string, e.g. "cautiously optimistic"), sentimentScore (number 0-100), and trends (array of objects, each with title, summary, impact one of "positive"|"neutral"|"negative", and category). Provide 4-6 trends.`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            overallSentiment: { type: Type.STRING },
            sentimentScore: { type: Type.NUMBER },
            trends: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  title: { type: Type.STRING },
                  summary: { type: Type.STRING },
                  impact: { type: Type.STRING },
                  category: { type: Type.STRING },
                },
                required: ["title", "summary", "impact", "category"],
              },
            },
          },
          required: ["overallSentiment", "sentimentScore", "trends"],
        },
      },
    });

    const parsed = JSON.parse(response.text || "{}");
    return {
      overallSentiment: parsed.overallSentiment || "Neutral",
      sentimentScore: typeof parsed.sentimentScore === "number" ? parsed.sentimentScore : 50,
      trends: Array.isArray(parsed.trends) ? parsed.trends : [],
      sources: [],
    };
  });
};

export const evaluateCareer = async (resume: ResumeData, targetRole: string): Promise<CareerMatch> => {
  return handleGeminiCall(async () => {
    const response = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: `Evaluate technical alignment for ${targetRole}: ${JSON.stringify(resume)}.`,
      config: { responseMimeType: "application/json" }
    });
    return JSON.parse(response.text || "{}");
  });
};

export const generateLearningPlan = async (gaps: string[], targetRole: string): Promise<LearningPlan> => {
  return handleGeminiCall(async () => {
    const response = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: `Generate a detailed 4-week learning roadmap for a ${targetRole} targeting these skill gaps: ${gaps.join(', ')}. For each week, provide: week number, topic title, 4-5 specific learning tasks, and 3-4 curated resources (with title, URL, and platform like "Udemy", "YouTube", "Documentation", etc.). Return valid JSON.`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            title: { type: Type.STRING },
            weeks: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  week: { type: Type.NUMBER },
                  topic: { type: Type.STRING },
                  tasks: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                  resources: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        title: { type: Type.STRING },
                        url: { type: Type.STRING },
                        platform: { type: Type.STRING },
                      },
                      required: ["title", "url", "platform"],
                    },
                  },
                },
                required: ["week", "topic", "tasks", "resources"],
              },
            },
          },
          required: ["title", "weeks"],
        },
      },
    });
    const parsed = JSON.parse(response.text || "{}");
    return {
      title: parsed.title || "Learning Roadmap",
      weeks: Array.isArray(parsed.weeks) ? parsed.weeks.map((w: any) => ({
        week: typeof w.week === "number" ? w.week : 0,
        topic: w.topic || "",
        tasks: Array.isArray(w.tasks) ? w.tasks : [],
        resources: Array.isArray(w.resources) ? w.resources.map((r: any) => ({
          title: r.title || "",
          url: r.url || "",
          platform: r.platform || "",
        })) : [],
      })) : [],
    };
  });
};

export const generateQuiz = async (topic: string, difficulty: string): Promise<QuizQuestion[]> => {
  return handleGeminiCall(async () => {
    const response = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: `Generate a 5-question ${difficulty} quiz on ${topic}. For each question, provide exactly 4 multiple choice options with one correct answer. Return valid JSON with this exact structure: {"questions":[{"id":"q1","question":"...","options":["opt1","opt2","opt3","opt4"],"correctAnswer":0,"explanation":"..."}]}. The correctAnswer MUST be a number (0, 1, 2, or 3).`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            questions: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  question: { type: Type.STRING },
                  options: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                  correctAnswer: { type: Type.NUMBER },
                  explanation: { type: Type.STRING },
                },
                required: ["id", "question", "options", "correctAnswer", "explanation"],
              },
            },
          },
          required: ["questions"],
        },
      },
    });
    const parsed = JSON.parse(response.text || "{}");
    const questions = Array.isArray(parsed.questions) ? parsed.questions : [];
    
    // Validate and normalize each question
    return questions.map((q: any, idx: number) => ({
      id: q.id || `q${idx + 1}`,
      question: q.question || "",
      options: Array.isArray(q.options) ? q.options.map(String) : [],
      correctAnswer: typeof q.correctAnswer === "number" ? Math.round(q.correctAnswer) : 0,
      explanation: q.explanation || ""
    }));
  });
};

export const assessPersonality = async (userName: string, answers: string[]): Promise<PersonalityAnalysis> => {
  return handleGeminiCall(async () => {
    const response = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: `You are a career psychologist. Analyze this person's behavioral DNA from their answers to 10 deep questions. Return a structured analysis.
Name: ${userName}
Answers (in order): ${answers.map((a, i) => `Q${i + 1}: ${a}`).join('\n')}

Provide: a short summary (3-4 bullet strings), 4-6 core strengths, 2-4 areas for development, 4-6 personality-matched careers, and 3-5 concrete career evolution steps. Set confidence 0-100 based on answer depth and specificity.`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            summary: { type: Type.ARRAY, items: { type: Type.STRING } },
            strengths: { type: Type.ARRAY, items: { type: Type.STRING } },
            development: { type: Type.ARRAY, items: { type: Type.STRING } },
            careers: { type: Type.ARRAY, items: { type: Type.STRING } },
            steps: { type: Type.ARRAY, items: { type: Type.STRING } },
            confidence: { type: Type.NUMBER },
          },
          required: ["summary", "strengths", "development", "careers", "steps", "confidence"],
        },
      },
    });
    const raw = response.text || "{}";
    const parsed = JSON.parse(raw);
    return {
      summary: Array.isArray(parsed.summary) ? parsed.summary : [],
      strengths: Array.isArray(parsed.strengths) ? parsed.strengths : [],
      development: Array.isArray(parsed.development) ? parsed.development : [],
      careers: Array.isArray(parsed.careers) ? parsed.careers : [],
      steps: Array.isArray(parsed.steps) ? parsed.steps : [],
      confidence: typeof parsed.confidence === "number" ? Math.min(100, Math.max(0, parsed.confidence)) : 0,
    };
  });
};

export const analyzePerformance = async (role: string, transcript: string): Promise<{ score: number, verdict: string, drawbacks: string[], actionableSteps: string[] }> => {
  return handleGeminiCall(async () => {
    const response = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: `Audit interview transcript for ${role}: ${transcript}`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            score: { type: Type.NUMBER },
            verdict: { type: Type.STRING },
            drawbacks: { type: Type.ARRAY, items: { type: Type.STRING } },
            actionableSteps: { type: Type.ARRAY, items: { type: Type.STRING } },
          },
          required: ["score", "verdict", "drawbacks", "actionableSteps"]
        }
      }
    });
    return JSON.parse(response.text || "{}");
  });
};

export const analyzeProjectCode = async (projectName: string, files: {name: string, content: string}[]): Promise<ProjectAnalysis> => {
  return handleGeminiCall(async () => {
    const fileSummary = files.map(f => `File: ${f.name}\n${f.content.substring(0, 500)}`).join('\n\n');
    const response = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: `Audit project "${projectName}":\n${fileSummary}\n\nProvide: technical complexity (0-10), code quality score (0-10), detected programming skills, architectural summary/critique, and 4-5 specific improvement suggestions.`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            projectName: { type: Type.STRING },
            technicalComplexity: { type: Type.NUMBER },
            qualityScore: { type: Type.NUMBER },
            detectedSkills: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            architecturalCritique: { type: Type.STRING },
            suggestions: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
          },
          required: ["projectName", "technicalComplexity", "qualityScore", "detectedSkills", "architecturalCritique", "suggestions"],
        },
      },
    });
    const parsed = JSON.parse(response.text || "{}");
    return {
      projectName: parsed.projectName || projectName,
      technicalComplexity: typeof parsed.technicalComplexity === "number" ? Math.min(10, Math.max(0, parsed.technicalComplexity)) : 0,
      qualityScore: typeof parsed.qualityScore === "number" ? Math.min(10, Math.max(0, parsed.qualityScore)) : 0,
      detectedSkills: Array.isArray(parsed.detectedSkills) ? parsed.detectedSkills : [],
      architecturalCritique: parsed.architecturalCritique || "",
      suggestions: Array.isArray(parsed.suggestions) ? parsed.suggestions : [],
    };
  });
};

export const processResume = async (t: string) => { const r = await initiateProfile(t); return { resumeData: r.resumeData, analysis: r.analysis }; };
export const rewriteResume = async (t: string) => { const r = await initiateProfile(t); return r.rewritten; };
