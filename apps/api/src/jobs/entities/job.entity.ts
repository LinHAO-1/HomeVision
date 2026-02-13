import {
  Entity,
  PrimaryColumn,
  Column,
  CreateDateColumn,
} from 'typeorm';

export type JobStatus = 'pending' | 'processing' | 'completed' | 'failed';

@Entity('jobs')
export class Job {
  @PrimaryColumn('uuid')
  id: string;

  @CreateDateColumn()
  created_at: Date;

  @Column({ type: 'varchar', length: 20, default: 'pending' })
  status: JobStatus;

  @Column({ type: 'jsonb', nullable: true })
  result_json: object | null;

  @Column({ type: 'text', nullable: true })
  error_message: string | null;
}
