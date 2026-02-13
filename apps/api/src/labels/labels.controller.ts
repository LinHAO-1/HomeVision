import {
  Controller,
  Get,
  Post,
  Put,
  Delete,
  Param,
  Body,
  Query,
  ParseIntPipe,
} from '@nestjs/common';
import { ApiTags, ApiOperation } from '@nestjs/swagger';
import { LabelsService, CreateLabelDto, UpdateLabelDto } from './labels.service';

@ApiTags('labels')
@Controller('api/v1/labels')
export class LabelsController {
  constructor(private readonly labelsService: LabelsService) {}

  @Post()
  @ApiOperation({ summary: 'Create or upsert a label for a photo' })
  async create(@Body() dto: CreateLabelDto) {
    return this.labelsService.create(dto);
  }

  @Get()
  @ApiOperation({ summary: 'Get all labels, or one by filename' })
  async findAll(@Query('filename') filename?: string) {
    if (filename) {
      const label = await this.labelsService.findByFilename(filename);
      return label ? [label] : [];
    }
    return this.labelsService.findAll();
  }

  @Get('export')
  @ApiOperation({ summary: 'Export all labels as JSON (for training)' })
  async exportAll() {
    return this.labelsService.exportAll();
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get a label by id' })
  async findOne(@Param('id', ParseIntPipe) id: number) {
    return this.labelsService.findOne(id);
  }

  @Put(':id')
  @ApiOperation({ summary: 'Update a label' })
  async update(
    @Param('id', ParseIntPipe) id: number,
    @Body() dto: UpdateLabelDto,
  ) {
    return this.labelsService.update(id, dto);
  }

  @Delete(':id')
  @ApiOperation({ summary: 'Delete a label' })
  async remove(@Param('id', ParseIntPipe) id: number) {
    await this.labelsService.remove(id);
    return { deleted: true };
  }
}
