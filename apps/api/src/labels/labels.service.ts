import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Label } from './entities/label.entity';
import { RoomType } from './entities/room-type.entity';

export interface CreateLabelDto {
  filename: string;
  roomType: string;
  amenities: string[];
  features: string[];
  modelPredictions?: object;
}

export interface UpdateLabelDto {
  roomType?: string;
  amenities?: string[];
  features?: string[];
}

interface LabelResponse {
  id: number;
  filename: string;
  roomType: string;
  amenities: string[];
  features: string[];
  modelPredictions: object | null;
  created_at: Date;
  updated_at: Date;
}

@Injectable()
export class LabelsService {
  constructor(
    @InjectRepository(Label)
    private labelsRepository: Repository<Label>,
    @InjectRepository(RoomType)
    private roomTypesRepository: Repository<RoomType>,
  ) {}

  private async resolveRoomType(name: string): Promise<RoomType> {
    let roomType = await this.roomTypesRepository.findOne({ where: { name } });
    if (!roomType) {
      roomType = this.roomTypesRepository.create({ name });
      roomType = await this.roomTypesRepository.save(roomType);
    }
    return roomType;
  }

  private toResponse(label: Label): LabelResponse {
    return {
      id: label.id,
      filename: label.filename,
      roomType: label.roomType?.name ?? 'Unknown',
      amenities: label.amenities,
      features: label.features,
      modelPredictions: label.modelPredictions,
      created_at: label.created_at,
      updated_at: label.updated_at,
    };
  }

  async create(dto: CreateLabelDto): Promise<LabelResponse> {
    const roomType = await this.resolveRoomType(dto.roomType);
    const existingLabel = await this.labelsRepository.findOne({
      where: { filename: dto.filename },
    });
    if (existingLabel) {
      existingLabel.roomType = roomType;
      existingLabel.amenities = dto.amenities;
      existingLabel.features = dto.features;
      existingLabel.modelPredictions = dto.modelPredictions ?? existingLabel.modelPredictions;
      existingLabel.updated_at = new Date();
      return this.toResponse(await this.labelsRepository.save(existingLabel));
    }
    const label = this.labelsRepository.create({
      filename: dto.filename,
      roomType,
      amenities: dto.amenities,
      features: dto.features,
      modelPredictions: dto.modelPredictions ?? null,
    });
    return this.toResponse(await this.labelsRepository.save(label));
  }

  async update(id: number, dto: UpdateLabelDto): Promise<LabelResponse> {
    const label = await this.labelsRepository.findOne({ where: { id } });
    if (!label) throw new NotFoundException('Label not found');
    if (dto.roomType !== undefined) {
      label.roomType = await this.resolveRoomType(dto.roomType);
    }
    if (dto.amenities !== undefined) label.amenities = dto.amenities;
    if (dto.features !== undefined) label.features = dto.features;
    label.updated_at = new Date();
    return this.toResponse(await this.labelsRepository.save(label));
  }

  async findAll(): Promise<LabelResponse[]> {
    const labels = await this.labelsRepository.find({ order: { id: 'ASC' } });
    return labels.map((label) => this.toResponse(label));
  }

  async findByFilename(filename: string): Promise<LabelResponse | null> {
    const label = await this.labelsRepository.findOne({ where: { filename } });
    return label ? this.toResponse(label) : null;
  }

  async findOne(id: number): Promise<LabelResponse> {
    const label = await this.labelsRepository.findOne({ where: { id } });
    if (!label) throw new NotFoundException('Label not found');
    return this.toResponse(label);
  }

  async remove(id: number): Promise<void> {
    await this.labelsRepository.delete(id);
  }

  async exportAll(): Promise<object[]> {
    const labels = await this.labelsRepository.find({ order: { id: 'ASC' } });
    return labels.map((label) => ({
      filename: label.filename,
      roomType: label.roomType?.name ?? 'Unknown',
      amenities: label.amenities,
      features: label.features,
    }));
  }
}
