import { Injectable, BadRequestException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';
import FormData from 'form-data';
import { Job } from './entities/job.entity';

const MAX_CONCURRENT_JOBS = 2;
const INFERENCE_URL = process.env.INFERENCE_URL || 'http://inference:8000';
const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
const ALLOWED_MIMES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];

let runningJobs = 0;

export interface PhotoResult {
  filename: string;
  roomType: { label: string; score: number; topPrompt: string };
  amenities: Array<{ label: string; score: number; prompt: string }>;
  features: Array<{ label: string; score: number; category: string; prompt: string }>;
  quality: {
    blurVar: number;
    brightness: number;
    width: number;
    height: number;
    isBlurry: boolean;
    isDark: boolean;
    overallScore: number;
  };
  adapterPredictions?: Array<{ label: string; confidence: number }>;
}

export interface JobResultsSummary {
  topAmenities: Array<{ label: string; count: number; avgScore: number }>;
  topFeatures: Array<{ label: string; category: string; count: number; avgScore: number }>;
  overallQualityScore: number;
}

@Injectable()
export class JobsService {
  constructor(
    @InjectRepository(Job)
    private jobRepo: Repository<Job>,
  ) {}

  async create(files: Express.Multer.File[]): Promise<{ jobId: string; status: string }> {
    if (!files?.length) {
      throw new BadRequestException({ message: 'At least 1 file required. Upload 1–20 photos.' });
    }
    if (files.length > 20) {
      throw new BadRequestException({ message: 'Maximum 20 files allowed. Upload 1–20 photos.' });
    }
    for (const f of files) {
      if (f.size > MAX_FILE_SIZE) {
        throw new BadRequestException({
          message: `File "${f.originalname}" exceeds 5MB limit.`,
        });
      }
      const mime = (f.mimetype || '').toLowerCase();
      if (!mime.startsWith('image/')) {
        throw new BadRequestException({
          message: `File "${f.originalname}" is not an image. Only image/* types allowed.`,
        });
      }
    }

    const id = uuidv4();
    const job = this.jobRepo.create({
      id,
      status: 'processing',
      result_json: null,
      error_message: null,
    });
    await this.jobRepo.save(job);

    setImmediate(() => this.runInference(id, files));
    return { jobId: id, status: 'processing' };
  }

  private async runInference(jobId: string, files: Express.Multer.File[]): Promise<void> {
    while (runningJobs >= MAX_CONCURRENT_JOBS) {
      await new Promise((r) => setTimeout(r, 500));
    }
    runningJobs += 1;
    try {
      const form = new FormData();
      for (const f of files) {
        form.append('files', f.buffer, { filename: f.originalname, contentType: f.mimetype });
      }
      const { data: photos } = await axios.post<PhotoResult[]>(
        `${INFERENCE_URL}/analyze/batch`,
        form,
        {
          headers: form.getHeaders(),
          maxBodyLength: Infinity,
          maxContentLength: Infinity,
        },
      );

      const summary = this.buildSummary(photos);
      const result_json = { summary, photos };
      await this.jobRepo.update(jobId, {
        status: 'completed',
        result_json,
        error_message: null,
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      await this.jobRepo.update(jobId, {
        status: 'failed',
        result_json: null,
        error_message: message,
      });
    } finally {
      runningJobs -= 1;
    }
  }

  private buildSummary(photos: PhotoResult[]): JobResultsSummary {
    const amenityScores: Record<string, number[]> = {};
    const featureData: Record<string, { scores: number[]; category: string }> = {};
    let qualitySum = 0;
    for (const p of photos) {
      qualitySum += p.quality.overallScore;
      for (const a of p.amenities) {
        if (!amenityScores[a.label]) amenityScores[a.label] = [];
        amenityScores[a.label].push(a.score);
      }
      for (const f of p.features ?? []) {
        if (!featureData[f.label]) featureData[f.label] = { scores: [], category: f.category };
        featureData[f.label].scores.push(f.score);
      }
    }
    const topAmenities = Object.entries(amenityScores).map(([label, scores]) => ({
      label,
      count: scores.length,
      avgScore: Math.round((scores.reduce((a, b) => a + b, 0) / scores.length) * 100) / 100,
    }));
    topAmenities.sort((a, b) => b.count - a.count);

    const topFeatures = Object.entries(featureData).map(([label, { scores, category }]) => ({
      label,
      category,
      count: scores.length,
      avgScore: Math.round((scores.reduce((a, b) => a + b, 0) / scores.length) * 100) / 100,
    }));
    topFeatures.sort((a, b) => b.count - a.count);

    const overallQualityScore =
      photos.length > 0
        ? Math.round((qualitySum / photos.length) * 100) / 100
        : 0;

    return { topAmenities, topFeatures, overallQualityScore };
  }

  async findOne(id: string): Promise<{
    id: string;
    status: string;
    results: { summary: JobResultsSummary; photos: PhotoResult[] } | null;
    errorMessage: string | null;
  }> {
    const job = await this.jobRepo.findOne({ where: { id } });
    if (!job) {
      throw new BadRequestException({ message: 'Job not found' });
    }
    return {
      id: job.id,
      status: job.status,
      results: job.result_json as { summary: JobResultsSummary; photos: PhotoResult[] } | null,
      errorMessage: job.error_message,
    };
  }
}
