import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
} from 'typeorm';

@Entity('labels')
export class Label {
  @PrimaryGeneratedColumn()
  id: number;

  @Column({ type: 'text' })
  filename: string;

  @Column({ type: 'text' })
  roomType: string;

  @Column({ type: 'jsonb' })
  amenities: string[];

  @Column({ type: 'jsonb' })
  features: string[];

  @Column({ type: 'jsonb', nullable: true })
  modelPredictions: object | null;

  @CreateDateColumn()
  created_at: Date;

  @Column({ type: 'timestamp', nullable: true })
  updated_at: Date;
}
