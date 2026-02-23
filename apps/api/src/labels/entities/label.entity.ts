import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  ManyToOne,
  JoinColumn,
} from 'typeorm';
import { RoomType } from './room-type.entity';

@Entity('labels')
export class Label {
  @PrimaryGeneratedColumn()
  id: number;

  @Column({ type: 'text' })
  filename: string;

  @ManyToOne(() => RoomType, { eager: true, nullable: false })
  @JoinColumn({ name: 'room_type_id' })
  roomType: RoomType;

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
