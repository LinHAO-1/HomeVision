import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { JobsModule } from './jobs/jobs.module';
import { LabelsModule } from './labels/labels.module';
import { Job } from './jobs/entities/job.entity';
import { Label } from './labels/entities/label.entity';

const dbUrl = process.env.DATABASE_URL;
const enableLabeling = process.env.ENABLE_LABELING === 'true';

const dbConfig = dbUrl
  ? {
      type: 'postgres' as const,
      url: dbUrl,
      ssl: { rejectUnauthorized: false },
      entities: [Job, Label],
      synchronize: true,
    }
  : {
      type: 'postgres' as const,
      host: process.env.DB_HOST || 'localhost',
      port: parseInt(process.env.DB_PORT || '5432', 10),
      username: process.env.DB_USERNAME || 'postgres',
      password: process.env.DB_PASSWORD || 'postgres',
      database: process.env.DB_DATABASE || 'homevision',
      entities: [Job, Label],
      synchronize: true,
    };

@Module({
  imports: [
    TypeOrmModule.forRoot(dbConfig),
    JobsModule,
    ...(enableLabeling ? [LabelsModule] : []),
  ],
})
export class AppModule {}
